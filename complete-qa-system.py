import os
from typing import List, Dict, Optional
from datetime import datetime
import hashlib
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pandas as pd
import docx
import plotly.express as px
import json
from io import BytesIO
import re
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Set up device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

class DocumentSegment:
    def __init__(self, content: str, source: str, page_num: Optional[int] = None):
        self.content = content
        self.source = source
        self.page_num = page_num

class LocalEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else \
                    "mps" if torch.backends.mps.is_available() else "cpu"
        
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            return embeddings.tolist()
        except Exception as e:
            raise

    def embed_query(self, text: str) -> List[float]:
        try:
            embedding = self.model.encode(
                [text],
                normalize_embeddings=True
            )
            return embedding[0].tolist()
        except Exception as e:
            raise

    def __call__(self, text: str | List[str]) -> List[List[float]] | List[float]:
        if isinstance(text, str):
            return self.embed_query(text)
        return self.embed_documents(text)

class DocumentProcessor:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.supported_formats = {
            "pdf": self._process_pdf,
            "docx": self._process_docx,
            "txt": self._process_txt
        }

    def _process_pdf(self, file) -> List[DocumentSegment]:
        segments = []
        reader = PdfReader(file)
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                segments.append(DocumentSegment(text, file.name, page_num))
        return segments

    def _process_docx(self, file) -> List[DocumentSegment]:
        doc = docx.Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return [DocumentSegment(text, file.name)]

    def _process_txt(self, file) -> List[DocumentSegment]:
        text = file.read().decode('utf-8')
        return [DocumentSegment(text, file.name)]

    def process_documents(self, files) -> List[DocumentSegment]:
        segments = []
        for file in files:
            file_extension = file.name.split('.')[-1].lower()
            if file_extension in self.supported_formats:
                segments.extend(self.supported_formats[file_extension](file))
        
        total_files = len(files)
        total_segments = len(segments)
        st.sidebar.info(f"Processed {total_files} documents, {total_segments} pages total")
        return segments

    def get_document_hash(self, segments: List[DocumentSegment]) -> str:
        content = "".join([seg.content for seg in segments])
        return hashlib.md5(content.encode()).hexdigest()

    def get_cached_embeddings(self, doc_hash: str, embeddings) -> Optional[FAISS]:
        cache_file = self.cache_dir / f"{doc_hash}.faiss"
        if cache_file.exists():
            try:
                return FAISS.load_local(
                    folder_path=str(cache_file),
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                st.warning(f"Cache loading failed: {str(e)}. Regenerating embeddings...")
                return None
        return None

    def cache_embeddings(self, doc_hash: str, vectorstore: FAISS):
        cache_file = self.cache_dir / f"{doc_hash}.faiss"
        vectorstore.save_local(str(cache_file))

    def clear_cache(self):
        try:
            for file in self.cache_dir.glob("*.faiss"):
                file.unlink()
            st.success("Cache cleared successfully")
        except Exception as e:
            st.error(f"Error clearing cache: {str(e)}")

class LocalLLM:
    def __init__(self, model_path: str = "./model"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def _chunk_context(self, context: List[DocumentSegment], max_length: int = 1024) -> str:
        formatted_segments = []
        current_length = 0
        
        for seg in context:
            formatted_seg = f"\n{seg.content}"
            tokens = len(self.tokenizer.encode(formatted_seg))
            
            if current_length + tokens <= max_length:
                formatted_segments.append(formatted_seg)
                current_length += tokens
            else:
                words = seg.content.split()
                half_length = len(words) // 2
                first_half = " ".join(words[:half_length])
                formatted_half = f"\n{first_half}"
                
                if current_length + len(self.tokenizer.encode(formatted_half)) <= max_length:
                    formatted_segments.append(formatted_half)
                break
        
        return "\n\n".join(formatted_segments)

    def get_response(self, context: List[DocumentSegment], query: str) -> str:
        try:
            chunked_context = self._chunk_context(context)
            
            prompt = f"""Use the following context to answer the question.
            If you cannot find the answer in the context, say "I cannot find relevant information in the provided context."
            Only use information from the context. Do not make up or infer information.
            
            Context:
            {chunked_context}

            Question: {query}

            Answer: """
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    num_beams=3,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    length_penalty=1.0,
                    early_stopping=True
                )
            
            response = self.tokenizer.decode(
                outputs[0, len(inputs["input_ids"][0]):],
                skip_special_tokens=True
            ).strip()
            
            return self._clean_response(response)

        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return "I apologize, but I encountered an error while processing your question. Please try rephrasing or ask another question."

    def _clean_response(self, response: str) -> str:
        response = re.sub(r'\s+', ' ', response)
        return response.strip()

class Analytics:
    @staticmethod
    def get_usage_stats(conversation_history: List[Dict]) -> pd.DataFrame:
        if not conversation_history:
            return pd.DataFrame()
        df = pd.DataFrame(conversation_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    @staticmethod
    def create_usage_charts(df: pd.DataFrame):
        if df.empty:
            return None, None

        daily_questions = (df.groupby(df['timestamp'].dt.date)
                         .size()
                         .reset_index(name='count'))
        fig1 = px.bar(daily_questions,
                     x='timestamp',
                     y='count',
                     title='Questions Per Day')

        fig2 = px.histogram(df,
                          x='response_length',
                          title='Response Length Distribution')

        return fig1, fig2

    @staticmethod
    def export_to_csv(conversation_history: List[Dict]) -> str:
        df = pd.DataFrame(conversation_history)
        return df.to_csv(index=False)

    @staticmethod
    def handle_export(conversation_history: List[Dict]):
        if 'export_type' not in st.session_state:
            st.session_state.export_type = None
            
        if 'export_data' not in st.session_state:
            st.session_state.export_data = None
            
        if st.button("Export to CSV"):
            st.session_state.export_type = "CSV"
            st.session_state.export_data = Analytics.export_to_csv(conversation_history)
        
        if st.session_state.export_type and st.session_state.export_data:
            st.download_button(
                "Download CSV",
                st.session_state.export_data,
                "conversation_export.csv",
                "text/csv"
            )

    @staticmethod
    def display_analytics(df: pd.DataFrame, container):
        if df.empty:
            container.info("No data available for analytics yet.")
            return

        fig1, fig2 = Analytics.create_usage_charts(df)
        
        if all([fig1, fig2]):
            container.plotly_chart(fig1, use_container_width=True)
            container.plotly_chart(fig2, use_container_width=True)
            
            container.subheader("Summary Statistics")
            stats = {
                "Total Questions": len(df),
                "Avg Response Length": f"{df['response_length'].mean():.0f} words"
            }
            
            stat_cols = container.columns(2)
            for i, (key, value) in enumerate(stats.items()):
                stat_cols[i].metric(key, value)

def main():
    load_dotenv()

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = None
    
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    if st.session_state.doc_processor is None:
        st.session_state.doc_processor = DocumentProcessor()
        
    if st.session_state.model is None:
        with st.spinner("Loading model... Please wait."):
            st.session_state.model = LocalLLM()
    
    if st.session_state.embeddings is None:
        with st.spinner("Loading embeddings model..."):
            st.session_state.embeddings = LocalEmbeddings(model_name="./local_all_mini_lm_v6")

    with st.sidebar:
        st.header("Settings & Upload")
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

        if st.button("Clear Conversation History"):
            st.session_state.conversation_history = []
            st.session_state.export_type = None
            st.session_state.export_data = None
            st.experimental_rerun()

        # Export section
        if st.session_state.conversation_history:
            st.header("Export Options")
            Analytics.handle_export(st.session_state.conversation_history)

    st.title("Document Question-Answering System")

    if uploaded_files:
        with st.spinner("Processing documents..."):
            segments = st.session_state.doc_processor.process_documents(uploaded_files)
            doc_hash = st.session_state.doc_processor.get_document_hash(segments)
            
            # Try to load from cache first
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = st.session_state.doc_processor.get_cached_embeddings(
                    doc_hash, 
                    st.session_state.embeddings
                )

            # If not in cache, create new embeddings
            if st.session_state.vectorstore is None:
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=500,
                    chunk_overlap=100,
                    length_function=len
                )
                
                texts = []
                metadatas = []
                for seg in segments:
                    chunks = text_splitter.split_text(seg.content)
                    texts.extend(chunks)
                    metadatas.extend([{
                        "source": seg.source,
                        "page": seg.page_num
                    }] * len(chunks))

                with st.spinner("Creating embeddings... This might take a while."):
                    st.session_state.vectorstore = FAISS.from_texts(
                        texts=texts,
                        embedding=st.session_state.embeddings,
                        metadatas=metadatas
                    )
                    st.session_state.doc_processor.cache_embeddings(
                        doc_hash, 
                        st.session_state.vectorstore
                    )

        st.header("Ask Questions")
        user_query = st.text_input("Enter your question:")

        if user_query:
            with st.spinner("Processing your question..."):
                start_time = datetime.now()
                
                # Get relevant documents with metadata
                docs = st.session_state.vectorstore.similarity_search(user_query, k=4)
                context_segments = []
                
                for doc in docs:
                    metadata = doc.metadata
                    context_segments.append(DocumentSegment(
                        content=doc.page_content,
                        source=metadata.get("source", "Unknown"),
                        page_num=metadata.get("page")
                    ))
                
                # Get response from local model
                response = st.session_state.model.get_response(
                    context_segments, 
                    user_query
                )
                processing_time = (datetime.now() - start_time).total_seconds()

                # Create new conversation entry
                new_conversation = {
                    'question': user_query,
                    'response': response,
                    'response_length': len(response.split()),
                    'timestamp': datetime.now(),
                    'processing_time': processing_time
                }

                st.session_state.conversation_history.append(new_conversation)

                # Display response section
                st.markdown("### Answer")
                st.write(response)
                
                # Display confidence and metrics
                metrics_cols = st.columns(4)
                # metrics_cols[0].metric("Confidence", f"{confidence:.2%}")
                metrics_cols[0].metric("Processing Time", f"{processing_time:.2f}s")
                metrics_cols[1].metric("Response Length", f"{len(response.split())} words")
                metrics_cols[2].metric("Total Queries", len(st.session_state.conversation_history))

                # # Display source documents used
                # st.markdown("### Source Documents")
                # with st.expander("View source segments used for this answer", expanded=False):
                #     for i, citation in enumerate(citations, 1):
                #         st.markdown(f"**Source {i}:** {citation['source']}" + 
                #                   (f" (Page {citation['page']})" if citation['page'] else ""))
                #         st.text(citation['content'])
                #         st.markdown("---")

        # Display Analytics
        st.header("Analytics")
        if st.session_state.conversation_history:
            df = Analytics.get_usage_stats(st.session_state.conversation_history)
            Analytics.display_analytics(df, st)
        else:
            st.info("Start asking questions to see analytics!")

        # Display conversation history
        if st.session_state.conversation_history:
            st.markdown("---")
            st.header("Conversation History")
            
            for i, conv in enumerate(reversed(st.session_state.conversation_history)):
                with st.expander(
                    f"Q{len(st.session_state.conversation_history)-i}: {conv['question'][:50]}...",
                    expanded=False
                ):
                    st.markdown("**Question:**")
                    st.write(conv['question'])
                    
                    st.markdown("**Response:**")
                    st.write(conv['response'])
                    
                    # Display citations if available
                    if conv.get('citations'):
                        st.markdown("**Sources:**")
                        for citation in conv['citations']:
                            source_text = f"- {citation['source']}"
                            if citation.get('page'):
                                source_text += f" (Page {citation['page']})"
                            st.markdown(source_text)
                    
                    cols = st.columns(4)
                    with cols[0]:
                        st.markdown("**Confidence:**")
                        st.info(f"{conv['confidence']:.2%}")
                    
                    with cols[1]:
                        st.markdown("**Response Length:**")
                        st.info(f"{conv['response_length']} words")
                    
                    with cols[2]:
                        st.markdown("**Processing Time:**")
                        st.info(f"{conv['processing_time']:.2f}s")
                    
                    with cols[3]:
                        st.markdown("**Time:**")
                        st.text(conv['timestamp'].strftime("%Y-%m-%d %H:%M:%S"))
    else:
        st.info("Please upload documents to start asking questions!")

if __name__ == "__main__":
    main()