from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationChain
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import FAISS
from langchain_experimental.agents import create_csv_agent
from google import genai
from google.genai import types

from typing import Dict, Any, Optional
import json
import requests
import os
import time
from urllib.parse import urlparse

# Note: You'll need to install and configure the video analysis client
# For now, we'll add a placeholder that returns a message
try:
    # Uncomment and configure these when you have the video analysis client set up
    # from your_video_client import client, MODEL_ID
    pass
except ImportError:
    print("Video analysis client not configured. Video analysis will return placeholder responses.")




class IntelligentToolRouter:
    def __init__(self, google_api_key: str):
        # ✅ Initialize LLM

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.3
        )

        # ✅ Initialize embeddings (Google Embeddings)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )

        # ✅ Create FAISS vector store (empty start, can preload with initial memory)
        vectorstore = FAISS.from_texts(["Initial memory seeded."], embeddings)

        # ✅ Retriever for memory
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # ✅ VectorStoreRetrieverMemory
        self.memory = VectorStoreRetrieverMemory(retriever=retriever)

        # ✅ Conversation Chain (with memory)
        self.conversation_chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )

        # ✅ Tools
        self.tools = {
            "question_query": {
                "function": self.Question_Query_agent,
                "description": "Answers questions about surveillance events data"
            },
            "video_analysis": {
                "function": self.Deep_video_search_agent,
                "description": "Analyzes video content from surveillance footage"
            },
            "graph_plotting": {
                "function": self.graph_plotting_agent,
                "description": "Creates visualizations of surveillance data"
            },
            "greeting_agent": {
                "function": self.greeting_agent,
                "description": "Greetings the user"
            }
        }

    # # === Router ===
    # def enhanced_user_query(self, user_query: str):
    #     """Enhance the user query to make it more specific and relevant"""

    #     prompt = f"""
    #     You are enhancing a user query for a surveillance events analysis system. The data contains:
        
    #     DATA CONTEXT:
    #     - Event Categories: Various surveillance events including vehicle control, behavioral analytics, and security incidents
    #     - Locations: Various camera locations across the surveillance network
    #     - Status: COMPLETED events
    #     - Media: Images and Videos with Google Cloud Storage URLs
    #     - Time Range: Events spanning multiple days/weeks with timestamps

        
    #     Original Query: "{user_query}"
        
    #     Enhanced Query (be specific about what data to analyze and how):
    #     """
        
    #     llm = ChatGoogleGenerativeAI(
    #         model="gemini-2.5-flash",
    #         google_api_key=self.llm.google_api_key,
    #         temperature=0
    #     )
    #     response = llm.invoke(prompt)
    #     return response.content


    def route_query(self, user_query: str, **kwargs) -> Dict[str, Any]:
        """Decide which tool to use, run it, and save results into memory"""
        routing_prompt = f"""
        Analyze the user query and select the appropriate tool:

        User Query: "{user_query}"

        Tools:
        1. question_query → Data Q&A, searches, filters, counts, video URL fetching , etc.
        2. video_analysis → Video content analysis
        3. graph_plotting → Graphs, plots, charts, visualizations

        Rules:
        - If query asks for charts, trends, plots → graph_plotting
        - If query is about video content/footage → video_analysis
        - If query is about greeting the user and other unrelated queries → greeting_agent
        - Else → question_query

        Respond with only the tool name.
        """

        # Ask LLM to decide tool
        response = self.llm.invoke(routing_prompt)
        selected_tool = response.content.strip().lower().replace("-", "_").replace(" ", "_")

        # Run tool
        if selected_tool in self.tools:
            result = self._execute_tool(selected_tool, user_query, **kwargs)
            # ✅ Save query + result into memory
            self.memory.save_context({"input": user_query}, {"output": str(result)})
            return {
                "selected_tool": selected_tool,
                "result": result,
                "success": True,
                "query": user_query
            }
        else:
            return {"error": f"Unable to route query. LLM selected: {selected_tool}"}

###########################################################################################################################

    def _execute_tool(self, tool_name: str, user_query: str, **kwargs) -> Any:
        if tool_name == "question_query":
            csv_file_path = kwargs.get("csv_file_path", "merged_events.csv")
            return self.Question_Query_agent(user_query, csv_file_path)

        elif tool_name == "video_analysis":
            video_url = kwargs.get("video_url")
            camera_id = kwargs.get("camera_id")
            return self.Deep_video_search_agent(user_query, video_url, camera_id)

        elif tool_name == "graph_plotting":
            csv_file_path = kwargs.get("csv_file_path", "merged_events.csv")
            return self.graph_plotting_agent(user_query, csv_file_path)

        elif tool_name == "greeting_agent":
            return self.greeting_agent(user_query)

###########################################################################################################################

    # === Tool Implementations ===
    def Question_Query_agent(self, user_enhanced_prompt, csv_file_path):


        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.llm.google_api_key,
            temperature=0.3
        )
        agent_executor = create_csv_agent(
            llm,
            csv_file_path,
            allow_dangerous_code=True
        )
        response = agent_executor.run(user_enhanced_prompt)
        return response

###########################################################################################################################


    def Deep_video_search_agent(self, user_enhanced_prompt:str, video_url:Optional[str]=None, camera_id:Optional[str]=None):
        """ This function is mostly for answering the query given by the user regarding video analysis """

        # Check if video_url is provided and is not a placeholder string
        if video_url and video_url != "string" and video_url.startswith(('http://', 'https://')):

            print("video_url=======>", video_url)
            try:
                MODEL_ID = "gemini-2.5-flash"
                GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyA30sppiuNaIdl3AoS4enXd3XMQ3Hi9jIo")
                client = genai.Client(api_key=GOOGLE_API_KEY)

                prompt = user_enhanced_prompt

                print("user_enhanced_prompt=======>", user_enhanced_prompt)
                parsed_url = urlparse(video_url)
                filename = os.path.basename(parsed_url.path)
                if not filename:
                    filename = f"video_{int(time.time())}.mp4"
                
                # Download video file
                response = requests.get(video_url)

                print("response=======>", response)
                with open(filename, 'wb') as f:
                    f.write(response.content)

                # Check if video analysis client is configured
                if client is None or MODEL_ID is None:
                    return f"Video analysis for '{user_enhanced_prompt}' on video {filename} would be performed here. Please configure your video analysis client to enable this functionality."

                # Use the video analysis client
                video_file = client.files.upload(file=filename)
                
                while video_file.state == "PROCESSING":
                    print('Waiting for video to be processed.')
                    time.sleep(10)
                    video_file = client.files.get(name=video_file.name)

                if video_file.state == "FAILED":
                    raise ValueError(video_file.state)

                response = client.models.generate_content(
                    model = MODEL_ID,
                    contents=[
                        video_file,
                        prompt
                    ]
                )

                print("final response=======>", response.text)

                return response.text
                
            except Exception as e:
                return f"Error processing video: {str(e)}"
        
        else:
            # Handle data queries about video URLs or camera information
            if "video url" in user_enhanced_prompt.lower() or "camera" in user_enhanced_prompt.lower():
                # This should be handled by the question_query tool instead
                return f"To find video URLs for cameras, please use the data query tool. This video analysis tool is for analyzing video content, not for retrieving video URLs from the database."
            else:
                return "No valid video URL provided. Please provide a proper video URL (starting with http:// or https://) for video analysis."

###########################################################################################################################



    # def graph_plotting_agent(self, user_enhanced_prompt, csv_file_path):
    #     llm = ChatGoogleGenerativeAI(
    #         model="gemini-2.5-flash",
    #         google_api_key=self.llm.google_api_key,
    #         temperature=0.3
    #     )
    #     agent_executor = create_csv_agent(
    #         llm,
    #         csv_file_path,
    #         allow_dangerous_code=True
    #     )
    #     response = agent_executor.run(user_enhanced_prompt)
    #     return response


    def graph_plotting_agent(self, user_enhanced_prompt, csv_file_path):
        import matplotlib
        from google.cloud import storage
        import random
        matplotlib.use("Agg")  # 🔒 force non-GUI backend (no windows pop up)

        import matplotlib.pyplot as plt  # must import AFTER backend set

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.llm.google_api_key,
            temperature=0.3
        )

        agent_executor = create_csv_agent(
            llm,
            csv_file_path,
            allow_dangerous_code=True
        )

        user_enhanced_prompt = user_enhanced_prompt+" "+"and save it in Variphi_generated_plot.png"

        print("user_enhanced_prompt======>",user_enhanced_prompt)

        response = agent_executor.run(user_enhanced_prompt)

        # Close figures to ensure files are flushed by matplotlib-driven code
        plt.close("all")

        # --- Find the locally saved PNG and rename it ---
        import glob
        import os
        # Prefer explicitly named file if the agent used it
        preferred = os.path.join(os.path.dirname(__file__), "Variphi_generated_plot.png")
        if os.path.exists(preferred):
            src_path = preferred
        else:
            # Fallback: pick the most recently modified PNG in this folder
            pngs = glob.glob(os.path.join(os.path.dirname(__file__), "*.png"))
            if not pngs:
                return {"llm_response": response, "error": "No PNG plot found to upload."}
            src_path = max(pngs, key=os.path.getmtime)

        random_number = random.randint(1000, 9999)
        file_name = f"Variphi_generated_plot_{random_number}.png"
        dst_path = os.path.join(os.path.dirname(__file__), file_name)
        try:
            os.replace(src_path, dst_path)
        except Exception as e:
            return {"llm_response": response, "error": f"Failed to rename plot: {e}"}

        # --- Upload to GCP bucket ---
        gcp_service_file = os.path.join(os.path.dirname(__file__), "generative-ai-variphi-098b98607a93.json")
        client = storage.Client.from_service_account_json(gcp_service_file)

        bucket_name = "vgidata"
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_filename(dst_path)

        # # === Public URL (if bucket is public) ===
        graph_url = f"https://storage.googleapis.com/{bucket_name}/{file_name}"

        # === Return both LLM response + URL ===
        return {
            "llm_response": response,
            "graph_url": graph_url,
            "filename": file_name
        }      
        # return response
#########################################################################################################################################


    






    # def graph_plotting_agent(self, user_enhanced_prompt, csv_file_path):
    #     import matplotlib
    #     matplotlib.use("Agg")  # 🔒 force non-GUI backend (no windows pop up)
    #     import matplotlib.pyplot as plt
    #     from google.cloud import storage
    #     import uuid
    #     import re

    #     # === Run LLM agent to generate graph code ===
    #     llm = ChatGoogleGenerativeAI(
    #         model="gemini-2.5-flash",
    #         google_api_key=self.llm.google_api_key,
    #         temperature=0.3
    #     )

    #     agent_executor = create_csv_agent(
    #         llm,
    #         csv_file_path,
    #         allow_dangerous_code=True
    #     )

    #     response = agent_executor.run(user_enhanced_prompt)


    #     # --- Save the current figure ---
    #     plot_file = "plot.png"
    #     plt.savefig(plot_file, bbox_inches='tight')  # ensures layout fits
    #     plt.close("all")  # free memory

    #     # --- Upload to GCP bucket ---
    #     gcp_service_file = "generative-ai-variphi-098b98607a93.json"
    #     client = storage.Client.from_service_account_json(gcp_service_file)



    #     bucket_name = "vgidata"
    #     bucket = nclient.bucket(bucket_name)
    #     blob = bucket.blob(plot_file)
    #     blob.upload_from_filename(plot_file)

    #     # # === Public URL (if bucket is public) ===
    #     graph_url = f"https://storage.googleapis.com/{bucket_name}/{plot_file}"

    #     # === Return both LLM response + URL ===
    #     return {
    #         "llm_response": response,
    #         "graph_url": graph_url
    #     }       


        # bucket = client.bucket(bucket_name)
        # blob = bucket.blob(plot_file)
        # blob.upload_from_filename(plot_file)
        # blob.make_public()  # optional: makes the file publicly accessible

        # plot_url = blob.public_url

        # return {
        #     "agent_response": response,
        #     "plot_url": plot_url
        # }

        # # === Extract filename from user prompt or create unique name ===
        # match = re.search(r"save it (\S+\.png)", user_enhanced_prompt)
        # if match:
        #     filename = match.group(1)
        # else:
        #     filename = f"graph_{uuid.uuid4().hex}.png"

        # # === Save locally ===
        # plt.savefig(filename, bbox_inches="tight")
        # plt.close("all")  # Free memory

        # # === Upload to GCP bucket ===
        # storage_client = storage.Client.from_service_account_json(
        #     "generative-ai-variphi-098b98607a93.json"
        # )
        # bucket_name = "vgidata"
        # bucket = storage_client.bucket(bucket_name)
        # blob = bucket.blob(filename)
        # blob.upload_from_filename(filename)

        # # === Public URL (if bucket is public) ===
        # graph_url = f"https://storage.googleapis.com/{bucket_name}/{filename}"

        # # === Return both LLM response + URL ===
        # return {
        #     "llm_response": response,
        #     "graph_url": graph_url,
        #     "filename": filename
        # }



        
    def greeting_agent(self, user_enhanced_prompt):
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.llm.google_api_key,
            temperature=0.1
        )
        response = llm.invoke(user_enhanced_prompt)
        return response.content