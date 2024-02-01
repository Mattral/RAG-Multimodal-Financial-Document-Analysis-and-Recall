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
](https://towardsdatascience.com/poppler-on-windows-179af0e50150)
[ Installing Tesseract on Windows
](https://indiantechwarrior.com/how-to-install-tesseract-on-windows/)


The process is simple after installing all the necessary packages and dependencies. We simply use the partition_pdf function, which extracts text and table data from the PDF and divides it into multiple chunks. We can customize the size of these chunks based on the number of characters.

```
from unstructured.partition.pdf import partition_pdf

raw_pdf_elements = partition_pdf(
    filename="./TSLA-Q3-2023-Update-3.pdf",
    # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
    # Titles are any sub-section of the document
    infer_table_structure=True,
    # Post processing to aggregate text once we have the title
    chunking_strategy="by_title",
    # Chunking params to aggregate text blocks
    # Attempt to create a new chunk 3800 chars
    # Attempt to keep chunks > 2000 chars
    # Hard max on chunks
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000
)
```

The previous code identifies and extracts various elements from the PDF, which can be classified into CompositeElements (the textual content) and Tables. We use the Pydantic package to create a new data structure that stores information about each element, including their type and text. The code below iterates through all extracted elements, keeping them in a list where each item is an instance of the Element type.

```
from pydantic import BaseModel
from typing import Any

# Define data structure
class Element(BaseModel):
    type: str
    text: Any

# Categorize by type
categorized_elements = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        categorized_elements.append(Element(type="table", text=str(element)))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        categorized_elements.append(Element(type="text", text=str(element)))
```

Creating the Element data structure enables convenient storage of the additional information, which can be beneficial for identifying the source of each answer, whether it is derived from texts, tables, or figures.

## 2. Graphs

The next step is gathering information from the charts to add context. The primary challenge is extracting images from the pages to feed into OpenAI's endpoint. A practical approach is to convert the PDF to images and pass each page to the model, inquiring if it detects any graphs. If it identifies one or more charts, the model can describe the data and the trends they represent. If no graphs are detected, the model will return an empty array as an indication.

💡
A drawback of this approach is that it increases the number of requests to the model, consequently leading to higher costs. The issue is that each page must be processed, regardless of whether it contains graphs, which is not an efficient approach. It is possible to reduce the cost by manually flagging the pages.

The initial step involves installing the pdf2image package to convert the PDF into images. This also requires the poppler tool, which we have already installed.

```
!pip install -q pdf2image==1.16.3
```
The code below uses the convert_from_path function, which takes the path of a PDF file. We can iterate over each page and save it as a PNG file using the .save() method. These images will be saved in the ./pages directory. Additionally, we define the pages_png variable that holds the path of each image.

```
import os
from pdf2image import convert_from_path


os.mkdir("./pages")
convertor = convert_from_path('./TSLA-Q3-2023-Update-3.pdf')

for idx, image in enumerate( convertor ):
    image.save(f"./pages/page-{idx}.png")

pages_png = [file for file in os.listdir("./pages") if file.endswith('.png')]
```

Defining a few helper functions and variables is necessary before sending the image files to the OpenAI API. The headers variable will contain the OpenAI API Key, enabling the server to authenticate our requests. The payload carries configurations such as the model name, the maximum token limit, and the prompts. It instructs the model to describe the graphs and generate responses in JSON format, addressing scenarios like encountering multiple graphs on a single page or finding no graphs at all. We will add the images to the payload before sending the requests. Finally, there is the encode_image() function, which encodes the images in base64 format, allowing them to be processed by OpenAI.

```
headers = {
  "Content-Type": "application/json",
  "Authorization": "Bearer " + str( os.environ["OPENAI_API_KEY"] )
}

payload = {
  "model": "gpt-4-vision-preview",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "You are an assistant that find charts, graphs, or diagrams from an image and summarize their information. There could be multiple diagrams in one image, so explain each one of them separately. ignore tables."
        },
        {
          "type": "text",
          "text": 'The response must be a JSON in following format {"graphs": [<chart_1>, <chart_2>, <chart_3>]} where <chart_1>, <chart_2>, and <chart_3> placeholders that describe each graph found in the image. Do not append or add anything other than the JSON format response.'
        },
        {
          "type": "text",
          "text": 'If could not find a graph in the image, return an empty list JSON as follows: {"graphs": []}. Do not append or add anything other than the JSON format response. Dont use coding "```" marks or the word json.'
        },
        {
          "type": "text",
          "text": "Look at the attached image and describe all the graphs inside it in JSON format. ignore tables and be concise."
        }
      ]
    }
  ],
  "max_tokens": 1000
}

# Function to encode the image to base64 format
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
```

The remaining steps include: 
1) utilizing the pages_png variable to loop through the images,
   2) encoding the image into base64 format,
      3) adding the image into the payload, and finally,
         4) sending the request to OpenAI and handling its responses.
            We will use the same Element data structure to store each image's type (graph) and the text (descriptions of the graphs).

```
