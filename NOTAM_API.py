# -*- coding: utf-8 -*-
"""
request NOTAM API
"""

def getNOTAM(api_key, states='', locations='') :

    import requests
    import json
    
    # Replace with the correct URL
    url = "https://v4p4sz5ijk.execute-api.us-east-1.amazonaws.com/anbdata/states/notams/notams-list"
    headers = {'api_key': api_key, 'states' : states, 'locations': locations}
    # It is a good practice not to hardcode the credentials. So ask the user to enter credentials at runtime
    myResponse = requests.get(url, params=headers, verify=True)
    #print (myResponse.status_code)
    
    print(myResponse)
    # For successful API call, response code will be 200 (OK)
    if(myResponse.ok):
    
        # Loading the response data into a dict variable
        # json.loads takes in only binary or string variables so using content to fetch binary content
        # Loads (Load String) takes a Json file and converts into python data structure (dict or list, depending on JSON)
        jData = json.loads(myResponse.content)
    
    
        error = myResponse.raise_for_status()
    else:
        jData = []
      # If response code is not ok (200), print the resulting http error code with description
        error = myResponse.raise_for_status()
    
    print(states, locations, ": The response contains {0} properties".format(len(jData)))
    #print("\n")
    
    return jData, error