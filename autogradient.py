import numpy as np    

class op:
    def dot(x1,x2,rec=True,bprop=False,grad=None):
        x1_val,x2_val = x1.get_value(),x2.get_value()
        #m,o   o,l      m,l                               
        if bprop==False:
            if rec:
                res = Tensor(np.dot(x1_val,x2_val),parent=[x1,x2],operation = [op.dot])
                x1.update_child(res)
                x2.update_child(res)
               
            else:
                res = np.dot(x1_val,x2_val)
            return res
            
        else:
            x1_grad = np.dot(grad,np.ascontiguousarray(x2_val.T))
            #                m,l,     l,o
            x2_grad = np.dot(np.ascontiguousarray(x1_val.T),grad)
            return x1_grad,x2_grad

    def add(x1,x2,rec=True,bprop=False,grad=None):
        x1_val,x2_val = x1.get_value(),x2.get_value()
        if bprop==False:
            if rec:
                res = Tensor(np.add(x1_val,x2_val),parent=[x1,x2],operation = [op.add])
                x1.update_child(res)
                x2.update_child(res)
            else:
                res = np.add(x1_val,x2_val)
            return res
        else:
            axis1 = op.__broadcast__(x1_val,grad)
            axis2 = op.__broadcast__(x2_val,grad)
            x1_grad = np.sum(grad , axis = axis1 ).reshape(x1_val.shape)
            x2_grad = np.sum(grad , axis = axis2 ).reshape(x2_val.shape)
            return x1_grad,x2_grad
    
    
    def sub(x1,x2,rec=True,bprop=False,grad=None):
        x1_val,x2_val = x1.get_value(),x2.get_value()
        if bprop==False:
            if rec:
                res = Tensor(np.subtract(x1_val,x2_val),parent=[x1,x2],operation = [op.sub])
                x1.update_child(res)
                x2.update_child(res)
            else:
                res = np.subtract(x1_val,x2_val)
            return res
        else:
            axis1 = op.__broadcast__(x1_val,grad)
            axis2 = op.__broadcast__(x2_val,grad)
            x1_grad = np.sum(grad , axis = axis1 ).reshape(x1_val.shape)
            x2_grad = np.sum(-1*grad , axis = axis2 ).reshape(x2_val.shape)
            return x1_grad,x2_grad
            
    def multiply(x1,x2,rec=True,bprop=False,grad=None):
        x1_val,x2_val = x1.get_value(),x2.get_value()
        if bprop==False:
            if rec:
                res = Tensor(np.multiply(x1_val,x2_val),parent=[x1,x2],operation = [op.multiply])
                x1.update_child(res)
                x2.update_child(res)
            else:
                res = np.multiply(x1_val,x2_val)
            return res
        else:
            axis1 = op.__broadcast__(x1_val,np.multiply(grad,x2_val))
            axis2 = op.__broadcast__(x2_val,np.multiply(grad,x1_val))
            x1_grad = np.sum(grad , axis = axis1 ).reshape(x1_val.shape)
            x2_grad = np.sum(grad , axis = axis2 ).reshape(x2_val.shape)
            
            return x1_grad,x2_grad
    def divide(x1,x2,rec=True,bprop=False,grad=None):
        x1_val,x2_val = x1.get_value(),x2.get_value()
        if bprop==False:
            if rec:
                res = Tensor(np.divide(x1_val,x2_val),parent=[x1,x2],operation = [op.divide])
                x1.update_child(res)
                x2.update_child(res)
            else:
                res = np.divide(x1_val,x2_val)
            return res
        else:
            axis1 = op.__broadcast__(x1_val,np.multiply(grad,x2_val))
            axis2 = op.__broadcast__(x2_val,np.multiply(grad,x1_val))
            
            x1_grad = np.sum(np.divide(grad,x2_val) , axis = axis1 ).reshape(x1_val.shape)
            x2_grad = np.sum(np.divide(np.multiply(grad,-1*x1_val),np.power(x2_val,2)) , axis = axis2 ).reshape(x2_val.shape)
            return x1_grad,x2_grad
    
    def crossentropy(x1,x2,rec=True,bprop=False,grad=None):
        x1_val,x2_val = x1.get_value(),x2.get_value()
        if bprop==False:
            res = np.sum(np.multiply(-1*np.log(x1_val),x2_val))
            if rec:
                res = Tensor(res,parent=[x1,x2],operation = [op.crossentropy])
                x1.update_child(res)
                x2.update_child(res)
            return res
        else:
            
            x1_grad = np.divide(-1*grad*x2_val,x1_val) 
            x2_grad = np.zeros(x2_val.shape)
            return x1_grad,x2_grad
    
    def mse(x1,x2,rec=True,bprop=False,grad=None):
        x1_val,x2_val = x1.get_value(),x2.get_value()
        if bprop==False:
            res = np.mean(np.subtract(x1_val,x2_val)**2)
            if rec:
                res = Tensor(res,parent=[x1,x2],operation = [op.mse])
                x1.update_child(res)
                x2.update_child(res)
            return res
        else:
            #axis1 = op.__broadcast__(x1_val,np.multiply(grad,x2_val))
            #axis2 = op.__broadcast__(x2_val,np.multiply(grad,x1_val))
            cach1 = 1
            cach2 = 1
            for a1 in x1_val.shape:
                cach1*=a1
            for a2 in x2_val.shape:
                cach2*=a2
            x1_grad = 2*np.subtract(x1_val,x2_val)*grad/max(cach1,cach2)
            x2_grad = 2*np.subtract(x2_val,x1_val)*grad/max(cach1,cach2)
            return x1_grad,x2_grad
    
        
    def exp(x1,rec=True,bprop=False,grad=None):
        x1_val = x1.get_value()
        if bprop==False:
            res = np.exp(x1_val)
            if rec:
                res = Tensor(res,parent=[x1],operation=[op.exp])
            return res # Note bprop doesnt matter since it is an expoential function
        else:
            res = np.exp(x1_val)
            return res#x1.get_consumer().get_value()
    
    def reshape(x1,shape,rec=True,bprop=False,grad=None):
        x1_val = x1.get_value()
        if bprop==False:
            res = np.reshape(x1_val,shape)
            if rec:
                res = Tensor(res,parent=[x1],operation = [op.reshape])
                x1.update_child(res)
            return res
        
        else:
            x1_grad = np.reshape(grad,x1_val.shape )
            return x1_grad
        
    
    
    def sum(x1,axis=None,rec=True,bprop=False,grad=None):
        # like np.sum this also doesnt axis
        # Issue: use keepdims in np.sum to retain dimensions
        x1_val = x1.get_value()
        if bprop==False:
            res = np.sum(x1_val,axis)
            if rec:
                res = Tensor(res,parent=[x1],operation = [op.sum])
                x1.update_child(res)
                x1.axis = axis # cache axis
            return res
        
        else:
            axis = x1.axis
            shp = list(x1_val.shape)
            if type(axis)==int:
                shp[axis] = 1

            else:
                for a in axis:
                    shp[a] = 1    
                    
            x1_grad[:] = np.reshape(grad,shp)
            return x1_grad
    
    def mean(x1,axis=None,rec= True,bprop=False,grad=None):
        # like np.sum this also doesnt axis
        x1_val = x1.get_value()
        if bprop==False:
            res =np.mean(x1_val,axis)
            if rec:
                res = Tensor(res,parent=[x1],operation = [op.mean,axis])
                x1.update_child(res)
                x1.axis = axis # cache: axis
            
            return res
        
        else:
            axis = x1.axis
            shp = list(x1_val.shape)
            cach = 1
            if type(axis)==int:
                shp[axis] = 1
                cach *= shp[axis]
            else:
                for a in axis:
                    cach *= shp[a]
                    shp[a] = 1
            
                 
            x1_grad[:] = np.reshape(grad,shp)/cach
            return x1_grad
    
    def softmax_crossentropy_loss(x1,x2,axis=None,rec=True,bprop=False,grad=None):
        x1_val,x2_val = x1.get_value(),x2.get_value()
        """Issue : Use keepdims in np.sum to prevent many errors """
        if bprop==False:
            
            
            shp = list(x1_val.shape)
            if type(axis)==int:
                shp[axis] = 1
            else:
                for a in axis:
                    shp[a] = 1
            # Taking care of overflow        
            x1_tmp = x1_val
            x1_tmpmax = np.max(x1_tmp,axis).reshape(shp)
            x1_tmp = x1_tmp-x1_tmpmax
            x1_tmp  = np.exp(x1_tmp)
                
            res = np.divide(x1_tmp,np.sum(x1_tmp,axis).reshape(shp))
            x1.cache  = res
            # adding 10**-20 to prevent error in log - numerically stable log
            res = np.sum(np.multiply(-1*np.log(res+10**-20),x2_val))
            
            if rec:
                res = Tensor(res,parent=[x1,x2],operation = [op.softmax_crossentropy_loss,axis])
                x1.update_child(res)
                x2.update_child(res)
            
            return res
        else:
            x1_grad = x1.cache-x2_val
            x2_grad = np.zeros(x2_val.shape)
            return x1_grad,x2_grad
        
    def softmax(x1,rec=True,axis=None):
        """ built using prebuilt functions """
        """Doesnt take care of overflow because it was built using prebuilt functions"""
        x1_tmp  = op.exp(x1,rec)
        res = op.divide(x1_tmp,op.sum(x1_tmp,axis,rec))
        
        return res
    
    def sigmoid(x1,rec=True,bprop=False,grad=None):
        x1_val = x1.get_value()
        if bprop==False:
            sigm = np.zeros(shape = x1_val.shape)
            sigm[x1_val>=0]= 1/(1.0 + np.exp(-x1_val[x1_val>=0]) )
            sigm[x1_val<0] = np.exp(x1_val[x1_val<0])/(1.0 + np.exp(x1_val[x1_val<0]) )
            x1.cache = sigm
                
            if rec:
                sigm = Tensor(sigm,parent=[x1],operation=[op.sigmoid])
                x1.update_child(sigm)
                
            return sigm
        else:
            sigm = x1.cache 
            return np.multiply(np.multiply(sigm,1.0-sigm),grad)
    
    def RelU(x1,rec=True,bprop=False,grad=None):
        x1_val = x1.get_value()
        if bprop == False:
            res = np.multiply(x1_val>0,x1_val)
            if rec:
                res  = Tensor(res,parent=[x1],operation=[op.RelU])
                x1.update_child(res)
                
            return res

        else:
            return np.multiply(x1_val>0,grad)
    
    def tanh(x1,rec=True,bprop=False,grad=None):
        x1_val = x1.get_value()
        if bprop == False:
            res = np.tanh(x1_val)
            if rec:
                res  = Tensor(res,parent=[x1],operation=[op.tanh])
                x1.update_child(res)
                
            return res

        else:
            return np.multiply((1 -np.power(np.tanh(x1_val),2)),grad)
            
            
    def __broadcast__(x1_val,grad1):
            x1_len = len(x1_val.shape)
            res_len = len(grad1.shape)
            res_shp = np.array(grad1.shape)
            x1_newsh = [1]*(res_len-x1_len)
            x1_newsh.extend(list(x1_val.shape))
            x1_newsh  = np.array(x1_newsh)
            return(tuple(np.where(x1_newsh-res_shp!=0)[0]))
            
"""    
Tensor.__add__ = op.add
Tensor.__sub__ = op.sub
Tensor.__mul__ = op.multiply
Tensor.__truediv__ = op.divide
Tensor.dot = op.dot
Tensor.reshape = op.reshape
Tensor.sum = op.sum
Tensor.mean = op.mean
Tensor.mse = op.mse
Tensor.exp = op.exp
Tensor.softmax_crossentropy_loss = op.softmax_crossentropy_loss
Tensor.softmax = op.softmax
Tensor.crossentropy = op.crossentropy
Tensor.sigmoid = op.sigmoid
Tensor.RelU = op.RelU
Tensor.tanh = op.tanh
"""




class Tensor:

    """
    Refer The below link to know why mutable datasetrutures cant be in the initialization
    https://stackoverflow.com/questions/4841782/python-constructor-and-default-value
    """
    def __init__(self,value,parent=None,child=None,operation=None,grad=None):
        
        self.value = np.array(value).astype(float)
        self.grad = grad
        self.operation = operation
        self.gradstatus = 0
        if child is None:
            self.child = []
        else:
            self.child = child
        if parent is None:
            self.parent = []
        else:
            self.parent = parent
            
        
    def get_value(self):
        return self.value
    def get_consumer(self):
        return self.child
    def get_parent(self):
        return self.parent
    def get_operation(self):
        return self.operation
    def get_grad(self):
        return self.grad
    def get_gradstatus(self):
        return self.gradstatus
    
    def update_child(self,child):
        self.child.append(child)
    def update_value(self,value):
        self.value = value
    def update_grad(self,grad):
        self.grad += grad
   
    def set_grad(self,grad):
        self.grad = grad
    def set_gradstatus(self,stat):
        self.gradstatus = stat
        
    def size(self):
        return 1
    def __repr__(self):
        return 'tensor object:'+ str(id(self)) 
        #return 'Tensor Value = ' +  str(self.value)
   #Additional methods
    __add__ = op.add
    __sub__ = op.sub
    __mul__ = op.multiply
    __truediv__ = op.divide
    dot = op.dot
    reshape = op.reshape
    sum = op.sum
    mean = op.mean
    mse = op.mse
    exp = op.exp
    softmax_crossentropy_loss = op.softmax_crossentropy_loss
    softmax = op.softmax
    crossentropy = op.crossentropy
    sigmoid = op.sigmoid
    RelU = op.RelU
    tanh = op.tanh


class graph:
    def __init__(self,func,var=None):
        self.func = func
        if var is None:
            self.var = []
        else:
            self.var = var
        self.func_anc = []
        self.var_desc = []
        self.gr = [] #final graph
        self.tmp = [] # stores cache nodes of desc and anc
    
    def get_ancestors(self,var):
        self.tmp=[]
        self.__get_ancestors__(var)
        return self.tmp
    
    def get_descendents(self,var):
        self.tmp=[]
        self.__get_descendents__(var)
        return self.tmp
    
    def get_graph(self):
        if len(self.gr) is 0:
            self.__get_graph__()
        return self.gr
    
    def get_grad(self):
        self.__get_graph__()   
        self.__grad_init__()
        self.__get_grad__(self.var)
        grad_table =[]
        for v in self.var:
            grad_table += [v.get_grad()]
        return grad_table
    
    def recompute_graph(self):
        self.__compute__(self.func)
        
    
    def __compute__(self,v):
        
        anc = v.get_parent()
        
        if len(anc)==1:
            if anc==None:
                return
            else:
                op = v.get_operation()
                if len(op)==1:
                    op = op[0]
                    self.__compute__(anc[0])
                    v.value = op(anc[0],rec=False)
                else:
                    axis = op[1]
                    op = op[0]
                    
                    self.__compute__(anc[0])
                    v.value = op(anc[0],axis=axis,rec=False)
        if len(anc)==2:
            op = v.get_operation()
            self.__compute__(anc[0])
            self.__compute__(anc[1])
            if len(op)==1:
                op = op[0]
                v.value = op(anc[0],anc[1],rec=False)
            else:
                axis = op[1]
                op = op[0]
                v.value = op(anc[0],anc[1],axis=axis,rec=False)
        
    
    def __get_grad__(self,var):
        for v in var:
            
            if v.get_gradstatus() is 0:
                
                child = self.__get_consumer__(v)
                
                for ch in child:
                    op = ch.get_operation()
                    pa = ch.get_parent()
                    self.__get_grad__([ch])
                    
                    if len(pa)==2:

                        pa1_grad_temp,pa2_grad_temp = op[0](x1 = pa[0],x2 = pa[1],bprop=True,grad=ch.get_grad())
                        if pa[0] == v:
                            v.update_grad(pa1_grad_temp)
                        elif pa[1] == v:
                            v.update_grad(pa2_grad_temp)
                    elif len(pa)==1:
                        pa1_grad_temp = op[0](x1 = pa[0],bprop = True,grad=ch.get_grad())
                        if pa[0] == v:
                            v.update_grad(pa1_grad_temp)
                        
            v.set_gradstatus(1)            
                    
    
    def __get_graph__(self):
        if len(self.gr) is 0:
            self.__var_union__()
            self.__func_anc__()
            self.gr = list(set(self.func_anc)&set(self.var_desc)|set(self.var)|set([self.func]))
        
        
    def __grad_init__(self):
        for nodes in self.gr:
            nodes.set_grad(0)
            nodes.set_gradstatus(0)
        self.func.set_grad(np.array([1.0]))
        self.func.set_gradstatus(1)

                    
    
    def __get_ancestors__(self,var):
        parent = var.get_parent()
        for p in parent:
            self.tmp.append(p)
            if p.size :
                self.__get_ancestors__(p)
               
    
    def __get_descendents__(self,var):
        child = var.get_consumer()
        for ch in child:
            self.tmp.append(ch)
            if ch.size:
                self.__get_descendents__(ch)
        
    
    def __var_union__(self):
        for v in self.var:
            self.var_desc = list(set(self.var_desc)|set(self.get_descendents(v)))
   
    def __func_anc__(self):
        self.func_anc = self.get_ancestors(self.func)
        
    def __get_consumer__(self,v):
        tmp_res = v.get_consumer()
        return (list(set(tmp_res)&set(self.gr)))
    
    def __get_parent__(self,v):
        tmp_res = v.get_parent()
        return (list(set(tmp_res)&set(self.gr)))

