# RAG-Multimodal-Financial-Document-Analysis-and-Recall


we will explore the application of the Retrieval-augmented Generation (RAG) method in processing a company's financial information contained within a PDF document. The process includes extracting critical data from a PDF file (like text, tables, graphs, etc.) and saving them in a vector store database such as Deep Lake for quick and efficient retrieval. Next, a RAG-enabled bot can access stored information to respond to end-user queries.

This task requires diverse tools, including [Unstructured.io](http://unstructures.io/) for text/table extraction, OpenAI's GPT-4V for extracting information from graphs, and LlamaIndex for developing a bot with retrieval capabilities. As previously mentioned, data preprocessing plays a significant role in the RAG process. So, we start by pulling data from a PDF document. The content of this repo focuses on demonstrating how to extract data from a single PDF document for ease of understanding. Nevertheless, the accompanying notebook provided after the repo will analyze three separate reports, offering a broader scope of information.

## Extracting Data
Extracting textual data is relatively straightforward, but processing graphical elements such as line or bar charts can be more challenging. The latest OpenAI model equipped with vision processing, GPT-4V, is valuable for visual elements. We can feed the slides to the model and ask it to describe it in detail, which then will be used to complement the textual information. This repo uses Tesla's [Q3 financial report](https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q3-2023-Update-3.pdf) as the source document. It is possible to download the document using the wget command.

```
wget https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q3-2023-Update-3.pdf
```
The preprocessing tasks outlined in the next section might be time-consuming and necessitate API calls to OpenAI endpoints, which come with associated costs. To mitigate this, we have made the preprocessed dataset and the checkpoints of the output of each section available at the end of this repo, allowing you to utilize them with the provided notebook.

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
graphs_description = []
for idx, page in tqdm( enumerate( pages_png ) ):
  # Getting the base64 string
  base64_image = encode_image(f"./pages/{page}")

  # Adjust Payload
  tmp_payload = copy.deepcopy(payload)
  tmp_payload['messages'][0]['content'].append({
    "type": "image_url",
    "image_url": {
      "url": f "data:image/png;base64,{base64_image}"
    }
  })

  try:
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=tmp_payload)
    response = response.json()
    graph_data = json.loads( response['choices'][0]['message']['content'] )['graphs']

    desc = [f"{page}\n" + '\n'.join(f"{key}: {item[key]}" for key in item.keys()) for item in graph_data]

    graphs_description.extend( desc )

  except:
    # Skip the page if there is an error.
    print("skipping... error in decoding.")
    continue;

graphs_description = [Element(type="graph", text=str(item)) for item in graphs_description]
```

## Store on Deep Lake
This section will utilize the Deep Lake vector database to store the collected information and their embeddings. These embedding vectors convert pieces of text into numerical representations that capture their meaning, enabling similarity metrics such as cosine similarity to identify documents with close relationships. For instance, a prompt inquiring about a company's total revenue would result in high cosine similarity with a database document stating the revenue amount as X dollars.

The data preparation is complete with the extraction of all crucial information from the PDF. The next step involves combining the output from the previous sections, resulting in a list containing 41 entries.


```
all_docs = categorized_elements + graphs_description

print( len( all_docs ) )
```

``
41
``

Given that we are using LlamaIndex, we can use its integration with Deep Lake to create and store the dataset. Begin by installing LlamaIndex and deeplake packages along with their dependencies.

```
!pip install -q llama_index==0.9.8 deeplake==3.8.8 cohere==4.37
```

Before using the libraries, it's essential to configure the OPENAI_API_KEY and ACTIVELOOP_TOKEN variables in the environment. Remember to substitute the placeholder values with your actual keys from the respective platforms.
```
import os

os.environ["OPENAI_API_KEY"] = "<Your_OpenAI_Key>"
os.environ["ACTIVELOOP_TOKEN"] = "<Your_Activeloop_Key>"
```

The integration of LlamaIndex enables the use of DeepLakeVectorStore class, which is designed to create a new dataset. Simply enter your organization ID, which by default is your Activeloop username, in the code provided below. This code will generate an empty dataset, ready to store documents.

```

from llama_index.vector_stores import DeepLakeVectorStore

# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "<YOUR-ACTIVELOOP-ORG-ID>"
my_activeloop_dataset_name = "tsla_q3"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

vector_store = DeepLakeVectorStore( dataset_path=dataset_path,
																		runtime={"tensor_db": True},
																		overwrite=False)
```

```
Your Deep Lake dataset has been successfully created!
```

Next, we must pass the created vector store to a StorageContext class. This class serves as a wrapper to create storage from various data types. In our case, we're generating the storage from a vector database, which is accomplished simply by passing the created database instance using the .from_defaults() method.

```
from llama_index.storage.storage_context import StorageContext

storage_context = StorageContext.from_defaults(vector_store=vector_store)
```

To store our preprocessed data, we must transform them into LlamaIndex Documents for compatibility with the library. The LlamaIndex Document is an abstract class that acts as a wrapper for various data types, including text files, PDFs, and database outputs. This wrapper facilitates the storage of valuable information with each sample. In our case, we can include a metadata tag to hold extra details like the data type (text, table, or graph) or denote document relationships. This approach simplifies the retrieval of these details later.

As shown in the code below, you can employ built-in classes like SimpleDirectoryReader to automatically read files from a specified path or proceed manually. It will loop through our list containing all the extracted information and assign text and a category to each document.

```
from llama_index import Document

documents = [Document(text=t.text, metadata={"category": t.type},) for t in categorized_elements]
```

Lastly, we can utilize the VectorStoreIndex class to generate embeddings for the documents and employ the database instance to store these values. By default, it uses OpenAI's Ada model to create the embeddings.
```
from llama_index import VectorStoreIndex

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
```

```
Uploading data to deeplake dataset.
100%|██████████| 29/29 [00:00<00:00, 46.26it/s]
\Dataset(path='hub://alafalaki/tsla_q3-nograph', tensors=['text', 'metadata', 'embedding', 'id'])

  tensor      htype      shape      dtype  compression
  -------    -------    -------    -------  ------- 
   text       text      (29, 1)      str     None   
 metadata     json      (29, 1)      str     None   
 embedding  embedding  (29, 1536)  float32   None   
    id        text      (29, 1)      str     None
```


The dataset has already been created and is hosted under the GenAI360 organization on the Activeloop hub. If you prefer not to use OpenAI APIs for generating embeddings, you can test the remaining codes using these publicly accessible datasets. Just substitute the dataset_path variable with the following: hub://genai360/tsla_q3.

## Get Deep Memory Access
As Step 0, please note that Deep Memory is a premium feature in Activeloop <b>paid plans</b>.

## Activate Deep Memory
The Deep Memory feature from Activeloop enhances the retriever's accuracy. This improvement allows the model to access higher-quality data, leading to more detailed and informative responses. In earlier repos, we already covered the basics of Deep Memory, so we will not dive into more details. The process begins by fetching chunks of data from the cloud and using GPT-3.5 to create specific questions for each chunk. These generated questions are then utilized in the Deep Memory training procedure to enhance the embedding quality. In our experience, this approach led to a 25% enhancement in performance.

Activeloop recommends using a dataset containing a minimum of 100 chunks, ensuring sufficient context for the model to enhance the embedding space effectively. So, the codes in this section are based on three PDF documents. For the complete code and execution steps to process three documents instead of one, please refer to the accompanying notebook.
The processed dataset is available in the cloud on the GenAI360 organization. You can access using the following key: hub://genai360/tesla_quarterly_2023.

The initial phase involves loading the pre-existing dataset and reading the text of each chunk along with its corresponding ID.

```
from llama_index.vector_stores import DeepLakeVectorStore

# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "<YOUR-ACTIVELOOP-ORG-ID>"
my_activeloop_dataset_name = "LlamaIndex_tsla_q3"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

db = DeepLakeVectorStore(
    dataset_path=dataset_path,
    runtime={"tensor_db": True},
    read_only=True
)

# fetch dataset docs and ids if they exist (optional you can also ingest)
docs = db.vectorstore.dataset.text.data(fetch_chunks=True, aslist=True)['value']
ids = db.vectorstore.dataset.id.data(fetch_chunks=True, aslist=True)['value']
print(len(docs))

```

```
Deep Lake Dataset in hub://genai360/tesla_quarterly_2023 already exists, loading from the storage

127
```

The following code segment outlines a function designed to use GPT-3.5 for generating questions corresponding to each data chunk. This involves crafting a specialized tool tailored for the OpenAI API. Primarily, the code configures suitable prompts for API requests to produce the questions and compiles them with their associated chunk IDs into a list.

```
import json
import random
from tqdm import tqdm
from openai import OpenAI

client = OpenAI()
# Set the function JSON Schema for openai function calling feature
tools = [
    {
        "type": "function",
        "function": {
            "name": "create_question_from_text",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question created from the given text",
                    },
                },
                "required": ["question"],
            },
            "description": "Create question from a given text.",
        },
    }
]


def generate_question(tools, text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "create_question_from_text"},
            },
            messages=[
                {"role": "system", "content": "You are a world class expert for generating questions based on provided context. You make sure the question can be answered by the text."},
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )

        json_response = response.choices[0].message.tool_calls[0].function.arguments
        parsed_response = json.loads(json_response)
        question_string = parsed_response["question"]
        return question_string
    except:
        question_string = "No question generated"
        return question_string

def generate_queries(docs: list[str], ids: list[str], n: int):

    questions = []
    relevances = []
    pbar = tqdm(total=n)
    while len(questions) < n:
        # 1. randomly draw a piece of text and relevance id
        r = random.randint(0, len(docs)-1)
        text, label = docs[r], ids[r]

        # 2. generate queries and assign and relevance id
        generated_qs = [generate_question(tools, text)]
        if generated_qs == ["No question generated"]:
            continue

        questions.extend(generated_qs)
        relevances.extend([[(label, 1)] for _ in generated_qs])
        pbar.update(len(generated_qs))

    return questions[:n], relevances[:n]

questions, relevances = generate_queries(docs, ids, n=20)
```

```
100%|██████████| 20/20 [00:19<00:00,  1.02it/s]
```

Now, we can use the questions and the reference ids to activate the Deep Memory using the .deep_memory.train() method to improve the embedding representations. You can see the status of the training process using the .info method.

```
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

job_id = db.vectorstore.deep_memory.train(
    queries=questions,
    relevance=relevances,
    embedding_function=embeddings.embed_documents,
)

print( db.vectorstore.dataset.embedding.info )
```

```
Starting DeepMemory training job
Your Deep Lake dataset has been successfully created!
Preparing training data for deepmemory:
Creating 20 embeddings in 1 batches of size 20:: 100%|██████████| 1/1 [00:03<00:00,  3.23s/it]
DeepMemory training job started. Job ID: 6581e3056a1162b64061a9a4

{'deepmemory': {'6581e3056a1162b64061a9a4_0.npy': {'base_recall@10': 0.25, 'deep_memory_version': '0.2', 'delta': 0.25, 'job_id': '6581e3056a1162b64061a9a4_0', 'model_type': 'npy', 'recall@10': 0.5}, 'model.npy': {'base_recall@10': 0.25, 'deep_memory_version': '0.2', 'delta': 0.25, 'job_id': '6581e3056a1162b64061a9a4_0', 'model_type': 'npy', 'recall@10': 0.5}}}
```

The dataset is now prepared and compatible with the Deep Memory feature. It's crucial to note that the Deep Memory option must be actively set to true when using the dataset for inference.


## Chatbot In Action
In this section, we will use the created dataset as the retrieval object, providing the necessary context for the GPT-3.5-turbo model (the default choice for LlamaIndex) to answer the questions. Keep in mind that the inference outcomes presented in the subsequent section are derived from processing three PDF files, which are consistent with the sample codes provided in the notebook. To access the processed dataset containing all the PDF documents, use hub://genai360/tesla_quarterly_2023 as the dataset path in the code below.

The DeepLakeVectorStore class also handles loading a dataset from the hub. The key distinction in the code below, compared to the previous sections, lies in the use of the .from_vector_store() method. This method creates indexes directly from the database rather than variables.

```
from llama_index.vector_stores import DeepLakeVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index import VectorStoreIndex

vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=False)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
)
```

We can now use the .as_query_engine() method of the index variables to establish a query engine. This will allow us to ask questions from various data sources. Notice the vector_store_kwargs argument, which activates the deep_memory feature by setting it to True. This step is essential for enabling the feature on the retriever. The .query() method takes a prompt and searches for the most relevant data points within the database to construct an answer.


```
query_engine = index.as_query_engine(vector_store_kwargs={"deep_memory": True})
response = query_engine.query(
    "What are the trends in vehicle deliveries?",
)
```
```
The trends in vehicle deliveries on the Quarter 3 report show an increasing trend over the quarters.
```

<img align="center" src="trend.avif" alt="banner">

As observed, the chatbot effectively utilized the data from the descriptions of the graphs we generated in the report. On the right, there's a screenshot of the bar chart which the chatbot referenced to generate its response.

Additionally, we conducted an experiment where we compiled the same dataset but excluded the graph descriptions. This dataset can be accessed via hub://genai360/tesla_quarterly_2023-nograph path. The purpose was to determine whether including the descriptions aids the chatbot's performance.

```
In quarter 3, there was a decrease in Model S/X deliveries compared to the previous quarter, with a 14% decline. However, there was an increase in Model 3/Y deliveries, with a 29% growth. Overall, total deliveries in quarter 3 increased by 27% compared to the previous quarter.
```

You'll observe that the chatbot points to incorrect text segments. Despite the answer being contextually similar, it doesn't provide the correct answer. The graph shows an upward trend, a detail that might not have been mentioned in the report's text.


