
def normalize(X):
    
    if type(X) == type([]):
        total = 0.0
        for val in X:
            total += val
        for i in range(len(X)):
            X[i] = X[i]/total
        return X
   
    if type(X) == type({}):
        total = 0.0
        for val in X.values():
            total += val
        for key in X:
            X[key] = X[key]/total
        return X
    return None
        
def UnionDict(d1, d2):
    dict = d1
    
    for k, v in d2.items():
        if k in dict:
            if type(v) != type(""):
                dict[k] = dict[k] + v
            else:
                dict[k] = v
        else:
            dict[k] = v
        
    return dict

if __name__=="__main__":
    print normalize({1:1,2:2,3:4})