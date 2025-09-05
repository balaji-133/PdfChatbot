
# 📘 Textbook Q\&A Chatbot

Upload a **Textbook**, ask any question, and get instant answers like:

* ✅ **2 Marks Answer** (short & precise)
* ✅ **8 Marks Answer** (detailed & explanatory)

This chatbot makes **studying from textbooks easier** by letting you interact with your notes, books, or study material directly.

## 🚀 Tech Stack & Tools

* 🟦 **[Groq AI](https://groq.com/)** → Ultra-fast inference for AI models
* 🟨 **[LangChain](https://www.langchain.com/)** → Framework for building question-answering pipelines
* 🟪 **[Hugging Face](https://huggingface.co/)** → Open-source LLMs and embeddings
* 🟩 **Python** → Backend & logic implementation
* 🟥 **Streamlit**) → Frontend for uploading textbooks and chatting

## ⚙️ Workflow (Step by Step)

### 1️⃣ Data Ingestion 📥

* Upload your **Textbook file**
* The bot extracts **text from the textbook**

### 2️⃣ Text Splitting ✂️

* Large textbooks are broken into **small chunks**
* Makes searching and retrieval faster


### 3️⃣ Embeddings 🔎

* Convert text chunks into **vector embeddings** using **Hugging Face models**
* Store in a **Vector Database**


### 4️⃣ Question Processing ❓

* User enters a question (e.g., *"Explain OS Deadlock in 8 marks"*)
* LangChain searches **relevant chunks** using embeddings

### 5️⃣ Answer Generation 📝

* Groq AI (fast LLM) generates answer:

  * 🎯 **2 Marks** → Short, direct answer
  * 📖 **8 Marks** → Detailed explanation with examples

### 6️⃣ Final Response 💬

* The chatbot displays both **short & long answers** in a clean format

## 🎨 Example Output

**Q:** *What is Operating System?*

* ✨ **2 Marks Answer:**
  An operating system is system software that manages hardware and software resources.

* ✨ **8 Marks Answer:**
  The operating system acts as an interface between hardware and users. It handles process management, memory management, file system operations, device handling, and provides security. Examples: Windows, Linux, macOS.

## 🌟 Why this Project?

* Helps **students** prepare for exams (2 marks & 8 marks style)
* Makes **textbooks interactive**
* Reduces **time wasted searching textbooks manually**


👉 Do you want me to make this README more **colorful with gradient-style headings** (multi-color effect using badges/HTML in markdown) so it pops on GitHub?
