#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from fei.model.utils import getLogger

#logger = getLogger()

# TODO: other be FeatureVector and CompositeFeatureVector??
# TODO: toString and load has to be changed, (tuple) -> float
# TODO: not sure if __mul__ and __add__ need to be overloaded for float
# TODO: possibly add self.add_amount to CompositeFeatureVector if __add__ to be enabled
# TODO: think about this one: feature_index (int) -> feature value (float)
# TODO: might have to maintain a feature indexing: string -> int


class FeatureVector(dict):
    """
    inherit from 'dict' class with overloaded operators
    """
    def __init__(self, *args, **kwargs):
        """
        unpack args and keyword args
        """
        super(FeatureVector, self).__init__(*args, **kwargs)
        pass
    
    def dot(self, other):
        """
        dot product of two vectors, return float
        """
        if not type(other) == FeatureVector: return
        (sht_vec, lng_vec) = (self, other) if len(self) < len(other) else (other, self)
        
        score = 0.0
        for k, v in sht_vec.iteritems(): 
            score += lng_vec.get(k, 0.0) * v
                
        return score
    
    def slice(self, other):
        """
        return new vector containing the same keys as in 'other' vector
        """
        if not type(other) == FeatureVector: return
        new_vec = FeatureVector()
        for k, _ in other.iteritems():
            new_vec[k] = self.get(k, 0.0)
            
        return new_vec
    
    def __iadd__(self, other):
        """
        overload operator '+=', merge keys in both vectors
        """
        if type(other) == FeatureVector: # handle FeatureVector
            for k, v in other.iteritems():
                self[k] = self.get(k, 0.0) + v
        
        if type(other) == CompositeFeatureVector: # handle CompositeFeatureVector
            feat_vec, scaling = other.feat_vec, other.scaling
            for k, v in feat_vec.iteritems():
                self[k] = self.get(k, 0.0) + v * scaling
        
        return self

    def __isub__(self, other):
        """
        overload operator '-=', merge keys in both vectors
        """
        # handle both FeatureVector and CompositeFeatureVector
        if type(other) == FeatureVector or type(other) == CompositeFeatureVector: 
            self += -1.0 * other
        
        return self
    
    def sub_cutoff(self, other, cutoff=0):
        """
        overload operator '-=', merge keys in both vectors
        """
        # handle both FeatureVector and CompositeFeatureVector
        if type(other) == FeatureVector: # handle FeatureVector
            for k, v in other.iteritems():
                self[k] = max(cutoff, self.get(k, 0.0) - v)
        
        if type(other) == CompositeFeatureVector: # handle CompositeFeatureVector
            feat_vec, scaling = other.feat_vec, other.scaling
            for k, v in feat_vec.iteritems():
                self[k] = max(cutoff, self.get(k, 0.0) - v * scaling)
        
        return self
    
    def __mul__(self, scaling):
        """
        overload operator '*'
        multiply feature vector with a scaling factor (float)
        """
        if not type(scaling) == float: return
        return CompositeFeatureVector(self, scaling)
    
    def __rmul__(self, scaling):
        """
        overload operator '*' (right-hand-side equivalent to __mul__)
        multiply feature vector with a scaling factor (float)
        """
        if not type(scaling) == float: return
        return CompositeFeatureVector(self, scaling)
    
    def toString(self):
        """
        output to string, assume keys are [tuples]
        """
        return '\n'.join(['%s\t%f' % (' '.join(k), v) for k, v in self.iteritems()])
    
    def load(self, input_file):
        """
        load FeatureVector using (feat, weight) pairs from file
        """
        import codecs
        import os.path
        
        if not os.path.exists(input_file):
            print 'file does not exist: %s' % input_file
        
        self.clear()
        with codecs.open(input_file, 'r', 'utf-8') as infile:
            for line in infile:
                line = line.strip()
                if line.startswith('#'): continue
                k, v = line.split('\t')
                new_k = tuple(k.split())
                self.setdefault(new_k, float(v))
                
        return self
    
    def save(self, output_file):
        """
        save FeatureVector using (feat, weight) pairs to file
        """
        import codecs
        import os.path
        
        if not os.path.exists(output_file):
            print 'file does not exist: %s' % output_file
        
        with codecs.open(output_file, 'w', 'utf-8') as outfile:
            outfile.writelines(self.toString())
        
        return self
    
    def getSquaredNorm(self):
        """
        calculate squared norm of feature values
        """
        return sum(v*v for v in self.itervalues())


class CompositeFeatureVector(object):
    """
    Couples a FeatureVector with a constant scaling factor
    """
    def __init__(self, feat_vec, scaling):
        self.feat_vec = feat_vec
        self.scaling = scaling
        return
    
    def __mul__(self, scaling):
        """
        overload operator '*'
        multiply feature vector with a scaling factor (float)
        """
        if not type(scaling) == float: return
        self.scaling *= scaling
        return self
    
    def __rmul__(self, scaling):
        """
        overload operator '*' (right-hand-side equivalent to __mul__)
        multiply feature vector with a scaling factor (float)
        """
        if not type(scaling) == float: return
        self.scaling *= scaling
        return self


if __name__ == '__main__':
    mydict1 = {('haha',):1, ('hehe',):2}
    mydict2 = {('hoho',):10, ('hiahia',):4, ('hehe',):2, ('lala',):1}
    
    
    feat_vec1 = FeatureVector(mydict1)
    feat_vec2 = FeatureVector(mydict2)
    
    print feat_vec1.sub_cutoff(feat_vec2)
    
    #feat_vec1 += feat_vec2 * (0.1 - 1)
    #print feat_vec1.toString()
    
    #print feat_vec1.save('log.txt')
    
    #print feat_vec1.load('log.txt')
    
    
    
    
