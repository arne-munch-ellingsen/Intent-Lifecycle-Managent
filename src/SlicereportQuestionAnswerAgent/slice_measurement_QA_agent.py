import os
from langchain_community.graphs import OntotextGraphDBGraph
from langchain_openai import ChatOpenAI
from langchain.chains import OntotextGraphDBQAChain

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "provide your open ai api key here"

# Explain the structure (template) of the GraphDB measurement graph
graph = OntotextGraphDBGraph(
    query_endpoint="http://Arnes-MacBook-Pro.local:7200/repositories/Slice1LatencyMeasurements",
    query_ontology="""
    PREFIX ex: <http://example.com/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX quan: <http://tio.models.tmforum.org/tio/v3.6.0/QuantityOntology/>

    CONSTRUCT {
        ex:Slice1 ex:hasLatencyMeasurement ?measurement .
        ?measurement rdf:type ex:LatencyMeasurement ;
        rdf:value ?value ;
        quan:unit ?unit ;
        prov:generatedAtTime ?timestamp .
    }
    WHERE {
        ex:Slice1 ex:hasLatencyMeasurement ?measurement .
        ?measurement rdf:type ex:LatencyMeasurement ;
        rdf:value ?value ;
        quan:unit ?unit ;
        prov:generatedAtTime ?timestamp .
    }
    """,  # The query to retrieve all measurements
)

# Create the QA chain
chain = OntotextGraphDBQAChain.from_llm(
    ChatOpenAI(temperature=0, model_name="gpt-4o"),
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True
)

# Interactive question loop
print("Welcome to the slice intent reports Latency Measurements QA agent. Type 'exit' to quit.")
while True:
    question = input("Enter your question: ")
    if question.lower() == "exit":
        print("Goodbye!")
        break
    try:
        response = chain.invoke(question)
        print(f"Response: {response}")
    except Exception as e:
        print(f"An error occurred: {e}")
