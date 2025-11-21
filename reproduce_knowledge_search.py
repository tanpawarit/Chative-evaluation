
import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from shared.knowledge_tools import knowledge_search
    
    print("Testing knowledge_search...")
    query = "ชื่อบริษัท"
    # You might need a valid workspace_id if the collection is partitioned
    workspace_id = "test"
    
    print(f"Query: {query}")
    print(f"Workspace ID: {workspace_id}")
    
    result = knowledge_search.invoke({"query": query, "workspace_id": workspace_id, "limit": 5})
    
    print("\nResult:")
    print(result)
    
    if result.get("chunks"):
        print("\nFirst chunk keys:")
        print(result["chunks"][0].keys())
        print("\nFirst chunk content:")
        print(result["chunks"][0])
    else:
        print("\nNo chunks found.")

except Exception as e:
    logger.exception("An error occurred:")
