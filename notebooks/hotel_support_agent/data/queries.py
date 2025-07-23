"""
Shared hotel search queries for both evaluation and testing.
Updated based on actual travel-sample data analysis.
"""

# Hotel search queries (based on travel-sample data)
HOTEL_SEARCH_QUERIES = [
    "Find hotels in Giverny with free breakfast",
    "I need a hotel in Glossop with free internet access",
    "Show me hotels in Helensburgh with free breakfast",
]

# Comprehensive reference answers matching actual database content
HOTEL_REFERENCE_ANSWERS = [
    # Query 1: Giverny with free breakfast
    """I found one hotel in Giverny that offers free breakfast:

**Le Clos Fleuri**
- **Location:** Giverny, France  
- **Address:** 5 rue de la Dîme, 27620 Giverny
- **Phone:** +33 2 32 21 36 51
- **Website:** http://www.giverny-leclosfleuri.fr/
- **Amenities:** Free breakfast ✅, Free internet ✅, Free parking ✅, No pets allowed
- **Vacancy:** Yes
- **Coordinates:** 49.0763077, 1.5234464
- **Reviews:** 3 customer reviews available with mixed ratings
- **Public Likes:** 7 likes
- **Description:** Situated near the church and just a few minutes walking distance from Monet's gardens and the Museum of Impressionisms, you will find Danielle and Claude's home, surrounded by a large magnificent garden, where you will find a haven of peace and tranquillity. Danielle speaks fluent English having spent many years in Australia.

This hotel is perfect for your stay in Giverny with the requested free breakfast amenity. It's ideally located for visiting Monet's gardens and offers a peaceful garden setting.""",
    # Query 2: Glossop with free internet
    """Here are hotels in Glossop that offer free internet access:

1. **The George Hotel**
   - **Address:** Norfolk Street, Glossop, United Kingdom
   - **Phone:** +44 1457 855449
   - **Price:** From £35.00 (single) or £60.00 (double)
   - **Amenities:** Free internet ✅, Free breakfast ✅, Pets allowed ✅
   - **Vacancy:** Yes
   - **Reviews:** 6 customer reviews available
   - **Coordinates:** 53.444331, -1.948299
   - **Description:** Set in the centre of town, this hotel makes an ideal base for a visit to the area.

2. **Avondale Guest House**
   - **Address:** 28 Woodhead Road, Glossop, United Kingdom
   - **Phone:** +44 1457 853132, Mobile: +44 7784 764969
   - **Website:** http://www.avondale-guesthouse.co.uk/
   - **Amenities:** Free internet ✅, Free breakfast ✅, Pets allowed ✅
   - **Vacancy:** Yes
   - **Reviews:** 7 customer reviews available
   - **Coordinates:** 53.449979, -1.945284

3. **The Bulls Head**
   - **Address:** 102 Church Street, Old Glossop, United Kingdom
   - **Phone:** +44 1457 866957, Mobile: +44 7876 744061, Restaurant: +44 1457 853291
   - **Website:** http://www.bulls-head.co.uk/
   - **Amenities:** Free internet ✅, Free breakfast ✅, Pets allowed ✅
   - **Vacancy:** No
   - **Reviews:** 1 customer review available
   - **Coordinates:** 53.450704, -1.939014
   - **Description:** Public House, Restaurant & Guest House.

4. **Windy Harbour Farm Hotel**
   - **Address:** Woodhead Road, Padfield, Glossop, United Kingdom
   - **Phone:** +44 1457 853107
   - **Website:** http://www.peakdistrict-hotel.co.uk/
   - **Amenities:** Free internet ✅, No free breakfast, No pets allowed
   - **Vacancy:** No
   - **Reviews:** 8 customer reviews available
   - **Coordinates:** 53.46327, -1.943125

These hotels are located in Glossop and the Glossop area, all offering the free internet access you're looking for.""",
    # Query 3: Helensburgh with free breakfast
    """Here are the hotels in Helensburgh that offer free breakfast:

1. **County Lodge Hotel**
   - **Location:** Helensburgh, United Kingdom
   - **Address:** Old Luss Road, Helensburgh, G84 7BH
   - **Phone:** +44 1436 672034
   - **Website:** http://www.countylodgehotel.co.uk/
   - **Amenities:** Free breakfast ✅, Free internet ✅, Free parking ✅, No pets allowed
   - **Price:** Rooms £40-£55
   - **Vacancy:** No
   - **Coordinates:** 55.99884, -4.71354
   - **Description:** Nearly 1 mile east of the town centre, near Colgrain Station.

2. **Commodore Hotel**
   - **Location:** Helensburgh, United Kingdom
   - **Address:** 112-117 West Clyde Street, Helensburgh, G84 8ES
   - **Phone:** +44 1436 676924
   - **Website:** http://www.innkeeperslodge.com/lodgedetail.asp?lid=91
   - **Amenities:** Free breakfast ✅, Free internet ✅, Pets allowed ✅, No free parking
   - **Price:** Rooms from £55
   - **Vacancy:** No
   - **Reviews:** 2 customer reviews available
   - **Coordinates:** 56.00481, -4.74472
   - **Description:** The biggest hotel in town with rooms from £55. Refurbished in about 2004. On the sea front about 1/2 mile from the town centre.

Both hotels offer the requested free breakfast along with additional amenities. The Commodore Hotel is described as the biggest hotel in town, refurbished in about 2004, and located on the sea front about 1/2 mile from the town centre.""",
]

# Create dictionary for backward compatibility
QUERY_REFERENCE_ANSWERS = {
    query: answer
    for query, answer in zip(HOTEL_SEARCH_QUERIES, HOTEL_REFERENCE_ANSWERS)
}


def get_evaluation_queries():
    """Get queries for evaluation"""
    return HOTEL_SEARCH_QUERIES


def get_all_queries():
    """Get all available queries"""
    return HOTEL_SEARCH_QUERIES


def get_simple_queries():
    """Get simple queries for basic testing"""
    return HOTEL_SEARCH_QUERIES


def get_reference_answer(query: str) -> str:
    """Get the correct reference answer for a given query"""
    return QUERY_REFERENCE_ANSWERS.get(
        query, f"No reference answer available for: {query}"
    )


def get_all_query_references():
    """Get all query-reference pairs"""
    return QUERY_REFERENCE_ANSWERS
