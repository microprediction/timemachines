from timemachines.skatertools.utilities.internet import connected_to_internet
if connected_to_internet():


    from timemachines.skatertools.recommendations.suggestions import top_rated_models, get_ratings
    from pprint import pprint


    def test_top_rated_models():
        for k in [1,5, 8, 13,21,34]:
           print('k='+str(k))
           suggestions = top_rated_models(k=k,max_seconds=500, require_passing=True)
           pprint(suggestions)
           print('')

    def dont_test_integrity_of_elo_ratings():
        PROPERTIES = ['name', 'count', 'rating', 'traceback', 'seconds', 'pypi']
        for category in ['univariate','residual']:
            for k in [1,2, 5, 8, 13,21,34]:
               print('k='+str(k))
               rd = get_ratings(k=k,category=category)
               for property in PROPERTIES:
                   if rd.get(property) is None:
                       raise ValueError('property '+property+' missing for category '+category+' k='+str(k))


if __name__=='__main__':
    assert connected_to_internet()
    dont_test_integrity_of_elo_ratings()