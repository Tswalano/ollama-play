from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional, List

class Settings(BaseSettings):
    # Directories
    DATA_DIR: Path = Path("./data")
    CHROMA_DIR: Path = Path("./chroma_db")

    # Database Configuration
    DB_URL: str = "sqlite:///conversations.db"

    # Model Configuration
    LLM_MODEL: str = "llama3.2:1b"
    EMBEDDING_MODEL: str = "mxbai-embed-large"
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 0.9
    LLM_TOP_K: int = 10
    LLM_REPEAT_PENALTY: float = 1.1
    LLM_STOP_SEQUENCES: List[str] = [
        "\nHuman:", "\nAssistant:", "Question:", "Context:", "Claude:", "If the human"
    ]

    # Application Settings
    FLASK_ENV: Optional[str] = "development"
    DEBUG: bool = True

    # RAG Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5

    # Prompts
    SYSTEM_PROMPT: str = (
        "You are an AI assistant focused on providing accurate information about company data. "
        "The company data includes:\n"
        "- Employee information (names, positions, salaries, departments)\n"
        "- Department details (budgets, locations, leadership)\n"
        "- Financial data (revenue, expenses, profits by quarter)\n\n"
        "Instructions:\n"
        "1. Use ONLY the information from the provided context.\n"
        "2. Be direct and specific in your responses.\n"
        "3. Include precise numbers when available.\n"
        "4. If information is not in the context, say so clearly.\n"
        "5. Format lists and data clearly.\n\n"
        "Context: {context}\n"
        "Question: {question}\n\nAnswer:"
    )
    FINANCIAL_PROMPT: str = (
        "Review the financial data and provide specific details:\n\n"
        "Data Guidelines:\n"
        "- Show exact revenue, expenses, and profit figures.\n"
        "- Include quarter and year references.\n"
        "- Present percentage changes when relevant.\n"
        "- Format numbers with proper currency symbols.\n\n"
        "Context: {context}\nQuestion: {question}\nAnswer:"
    )
    EMPLOYEE_PROMPT: str = (
        "Provide employee information with the following details:\n\n"
        "Required Information:\n"
        "- Full name and position.\n"
        "- Department and location.\n"
        "- Salary and hire date.\n"
        "- Reporting structure (if applicable).\n\n"
        "Context: {context}\nQuestion: {question}\nAnswer:"
    )
    DEPARTMENT_PROMPT: str = (
        "Present department information including:\n\n"
        "Key Details:\n"
        "- Department name and location.\n"
        "- Budget allocation.\n"
        "- Team size and structure.\n"
        "- Key performance metrics.\n\n"
        "Context: {context}\nQuestion: {question}\nAnswer:"
    )

    # Environment Configuration
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"

    def get_prompt_for_type(self, query_type: str) -> str:
        """Retrieve the appropriate prompt template based on query type."""
        return {
            'financial': self.FINANCIAL_PROMPT,
            'employee': self.EMPLOYEE_PROMPT,
            'department': self.DEPARTMENT_PROMPT,
            'general': self.SYSTEM_PROMPT,
        }.get(query_type, self.SYSTEM_PROMPT)

# Initialize settings
settings = Settings()