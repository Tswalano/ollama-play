import pandas as pd
import numpy as np
from typing import List, Dict, Any
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from app.utils.logger import logger
from app.config import Settings
import re


class RAGManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = OllamaEmbeddings(model=settings.EMBEDDING_MODEL)
        self.llm = Ollama(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            top_p=settings.LLM_TOP_P,
            top_k=settings.LLM_TOP_K,
            repeat_penalty=settings.LLM_REPEAT_PENALTY,
            stop=settings.LLM_STOP_SEQUENCES,
        )
        self.vectorstore = None
        self.pandas_ai = PandasAI(OpenAI(api_token=settings.OPENAI_API_KEY))
        self.dataframe = None
        self._initialize_rag_system()

    def _initialize_rag_system(self):
        """Initialize the RAG system by loading and preparing data."""
        try:
            self.dataframe = self._load_and_merge_data()
            self.vectorstore = Chroma.from_texts(
                texts=self._format_documents(self.dataframe),
                embedding=self.embeddings,
                persist_directory=str(self.settings.CHROMA_DIR),
            )
            logger.info("RAG system initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing RAG: {e}")
            raise

    def _load_and_merge_data(self) -> pd.DataFrame:
        """Load and merge all datasets into a single DataFrame."""
        try:
            employees = pd.read_csv(self.settings.DATA_DIR / "employees.csv")
            departments = pd.read_csv(self.settings.DATA_DIR / "departments.csv")
            financials = pd.read_csv(self.settings.DATA_DIR / "financials.csv")

            # Merge datasets
            emp_dept = pd.merge(employees, departments, left_on="department_id", right_on="id", suffixes=('_emp', '_dept'))
            full_data = pd.merge(emp_dept, financials, left_on="department_id", right_on="department_id", how="left")
            return full_data
        except Exception as e:
            logger.error(f"Error loading and merging data: {e}")
            raise

    def _format_documents(self, data: pd.DataFrame) -> List[str]:
        """Format the combined DataFrame into a list of document strings."""
        try:
            return [
                f"Employee: {row['first_name']} {row['last_name']} ({row['position']}), "
                f"Department: {row['name_dept']} (Location: {row['location']}), "
                f"Salary: ${row['salary']:,}, Budget: ${row['budget']:,}, "
                f"Financials: Q{row['quarter']} {row['year']}, Revenue: ${row['revenue']:,}, "
                f"Expenses: ${row['expenses']:,}, Profit: ${row['profit']:,}"
                for _, row in data.iterrows()
            ]
        except Exception as e:
            logger.error(f"Error formatting documents: {e}")
            raise

    def query(self, question: str) -> str:
        """Query the unified DataFrame using Pandas AI."""
        try:
            result = self.pandas_ai.run(self.dataframe, question)
            return self._format_response(result)
        except Exception as e:
            logger.error(f"Error querying data: {e}")
            raise

    def _format_response(self, response: str) -> str:
        """Format the response for presentation."""
        try:
            response = re.sub(r'\$(\d+)', lambda m: f"${int(m.group(1)):,}", response)
            return response.strip()
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            raise
