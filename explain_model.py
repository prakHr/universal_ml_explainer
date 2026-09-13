from lightshap import explain_any, explain_tree
# from sklearn.linear_model import BayesianRidge
import numpy as np
from pprint import pprint
    



def explain_models(X,y,model):
    X = np.array(X)
    y = np.array(y)
    model.fit(X, y)
    explanation = explain_any(model.predict, X)
    results = {
        "Feature importance":explanation.plot.bar(),          
        "Summary plot":explanation.plot.beeswarm(),           
        "Dependence plots":explanation.plot.scatter(),        
        "Individual explanation":explanation.plot.waterfall() 
    }
    return results


     

# if __name__=="__main__":
        
#     X = [
#         [1, 2],
#         [2, 3],
#         [3, 4],
#         [4, 5],
#         [5, 6],
#         [6, 7],
#         [7, 8],
#         [8, 9],
#         [9, 10],
#         [10, 11],
#         [11, 12],
#         [12, 13],
#         [13, 14],
#         [14, 15],
#         [15, 16],
#         [16, 17],
#         [17, 18],
#         [18, 19],
#         [19, 20],
#         [20, 21],
#         [21, 22],
#     ]

#     y = [
#         0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0,
#         1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1,
#         1
#     ]
    
#     model = BayesianRidge()
#     results = explain_models(X,y,model)
#     pprint(results)
