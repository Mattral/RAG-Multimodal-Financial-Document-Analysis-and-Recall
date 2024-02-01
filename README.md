# RAG-Multimodal-Financial-Document-Analysis-and-Recall


we will explore the application of the Retrieval-augmented Generation (RAG) method in processing a company's financial information contained within a PDF document. The process includes extracting critical data from a PDF file (like text, tables, graphs, etc.) and saving them in a vector store database such as Deep Lake for quick and efficient retrieval. Next, a RAG-enabled bot can access stored information to respond to end-user queries.

This task requires diverse tools, including [Unstructured.io](http://unstructures.io/) for text/table extraction, OpenAI's GPT-4V for extracting information from graphs, and LlamaIndex for developing a bot with retrieval capabilities. As previously mentioned, data preprocessing plays a significant role in the RAG process. So, we start by pulling data from a PDF document. The content of this lesson focuses on demonstrating how to extract data from a single PDF document for ease of understanding. Nevertheless, the accompanying notebook provided after the lesson will analyze three separate reports, offering a broader scope of information.

## Extracting Data
Extracting textual data is relatively straightforward, but processing graphical elements such as line or bar charts can be more challenging. The latest OpenAI model equipped with vision processing, GPT-4V, is valuable for visual elements. We can feed the slides to the model and ask it to describe it in detail, which then will be used to complement the textual information. This lesson uses Tesla's [Q3 financial report](https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q3-2023-Update-3.pdf) as the source document. It is possible to download the document using the wget command.

```
wget https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q3-2023-Update-3.pdf
```
The preprocessing tasks outlined in the next section might be time-consuming and necessitate API calls to OpenAI endpoints, which come with associated costs. To mitigate this, we have made the preprocessed dataset and the checkpoints of the output of each section available at the end of this lesson, allowing you to utilize them with the provided notebook.

## 1. Text/Tables
The unstructured package is an effective tool for extracting information from PDF files. It requires two tools, popplerand tesseract, that help render PDF documents. We suggest setting up these packages on Google Colab, freely available for students to execute and experiment with code. We will briefly mention the installation of these packages on other operating systems. Let's install the utilities and their dependencies using the following commands.

```
apt-get -qq install poppler-utils
apt-get -qq install tesseract-ocr

pip install -q unstructured[all-docs]==0.11.0 fastapi==0.103.2 kaleido==0.2.1 uvicorn==0.24.0.post1 typing-extensions==4.5.0 pydantic==1.10.13
```

These packages are easy to install on Linux and Mac operating systems using apt-get and brew. However, they are more complex to install on Windows OS. You can follow the below instructions for a step-by-step guide if you use Windows. 
[ Installing Poppler on Windows 
](https://towardsdatascience.com/poppler-on-windows-179af0e50150)[ Installing Tesseract on Windows
](https://indiantechwarrior.com/how-to-install-tesseract-on-windows/)https://indiantechwarrior.com/how-to-install-tesseract-on-windows/]
