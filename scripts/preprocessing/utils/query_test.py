

from SPARQLWrapper import SPARQLWrapper, JSON, GET
import time

endpoint = "https://query.wikidata.org/sparql"
datastore = SPARQLWrapper(endpoint=endpoint)
datastore.setTimeout(5)
datastore.setReturnFormat(JSON)
datastore.setMethod(GET)

while True:

    datastore.setQuery("SELECT * WHERE {?s ?p ?o} LIMIT 1")
    try:
        response = datastore.queryAndConvert()
    except Exception as e:
        print(e)
        time.sleep(5)
        continue
    print(response)