# ref: Pecora, L.M.: Cluster synchronization and isolated desynchronization in complex networks with symmetries. Nat. Commun. 5, 4079 (2014).
# 使用SageMath运行环境，需要安装SageMath
# 下载地址：https://www.sagemath.org/download.html

import scipy as sp
import numpy.random as rn
import numpy.linalg as la
import numpy as np
import sage.all as sg

class NetworkIRR:
    '''
        类 NetworkIRR:
        
            NetworkIRR 对象包含一个表示网络的邻接矩阵，以及与该网络相关的 IRR 坐标系统的数据。
        
        实例数据：
        
            _adjmat: 节点坐标中的邻接矩阵
            _nnodes: 整数，节点数量
            _group: SageGroup 对象，网络的自同构群
            _permutation_matricies: 表示自同构群中元素的节点空间置换矩阵的列表
            _conjugacy_classes: Sage 调用的结果
            _conjugacy_classes_matrices: 表示共轭类的矩阵列表，顺序与 _conjugacy_classes 一致
            _character_table: Sage 调用的结果
            _IRR_degeneracies: 每个 IRR 在置换矩阵表示中出现的次数，顺序与字符表中的 IRR 顺序相同
            _projection_operators: 表示每个 IRR 子空间投影算子的列表
            _transformation_operator: 转换矩阵
    '''
    def __init__(self, adjmat=None):
        '''
            NetworkIRR 对象的构造函数。 
            
            参数：
                adjmat: 邻接矩阵，2维 numpy 数组
        '''
        self._reset_data()
        self._set_adjacency_matrix(adjmat)
        
    def _reset_data(self):
        ''' 
            重置 NetworkIRR 对象的所有数据。
            在调用 _set_adjacency_matrix 和构造函数时调用此方法。
            这是一个私有方法，不建议直接调用
        '''
        self._adjmat=None
        self._nnodes=0
        self._group=None
        self._orbits=None
        self._permutation_matrices=None
        self._conjugacy_classes=None
        self._conjugacy_classes_matrices=None
        self._character_table=None
        self._IRR_degeneracies=None
        self._projection_operators=None
        self._transformation_operator=None

    def _set_adjacency_matrix(self,adjmat):
        '''
            设置邻接矩阵。 
            
            注意：调用此方法会清除所有数据（即群元素，投影算子等）。
            这是一个私有方法，不建议直接调用。
        '''
        self._reset_data()
        self._adjmat=np.array(adjmat.copy())
        self._nnodes=len(self._adjmat)
        
    def get_adjacency_matrix(self):
        '''
            返回当前邻接矩阵的深拷贝。 
        '''
        return self._adjmat.copy()
        
    def get_automorphism_group(self):
        '''
            返回网络的自同构群的深拷贝。
            
            使用 Sage 创建的群对象。  
        '''
        if self._group==None:
            self._group,self._orbits = \
            (sg.Graph(sg.Matrix(self._adjmat))).automorphism_group(orbits=True)
        
        return sg.copy(self._group)
        
    def get_automorphism_group_matrices(self):
        '''
        返回自同构群的所有置换矩阵（每个矩阵为 N x N 的 numpy 矩阵）
        
        这些是节点空间中的置换矩阵，表示群中的元素。使用此方法时要小心，
        如果网络较大，包含大量对称性，结果可能会占用大量内存。
        '''
        if self._group==None:
            self.get_automorphism_group()
        
        if self._permutation_matrices==None:
            self._permutation_matrices=[]
            
            for element in self._group.list():
                self._permutation_matrices\
                .append(np.array(element.matrix()))
                
        return list(self._permutation_matrices)
        
    def get_orbits(self):
        '''
            返回自同构群的轨道，作为包含节点的列表
            
            轨道是指在自同构群下可以互换的节点集合。
        '''
        if self._orbits==None:
            self._group,self._orbits = sg.Graph(sg.Matrix(self._adjmat)).automorphism_group(orbits=True)
            
        return sg.copy(self._orbits)
    
    def get_character_table(self):
        '''
            返回自同构群的字符表
            
            字符表提供了自同构群表示的具体信息。
        '''
        if self._character_table==None:
            self._character_table=self.get_automorphism_group().character_table()

        return sg.copy(self._character_table)
    
    def get_conjugacy_classes(self):
        '''
            返回自同构群的共轭类
            
            共轭类是具有相同结构特征的群元素的集合。
        '''
        if self._conjugacy_classes==None:
            if self._group==None:
                self.get_automorphism_group()
                
            self._conjugacy_classes=self._group.conjugacy_classes()
            
        return sg.copy(self._conjugacy_classes)
        
    def get_conjugacy_classes_matrices(self):
        '''
        返回共轭类对应的置换矩阵列表
        
        每个共轭类对应一个包含矩阵的子列表，这些矩阵在节点空间中表示。
        '''
        if self._conjugacy_classes==None:
            self.get_conjugacy_classes()
        
        if self._conjugacy_classes_matrices==None:
            self._conjugacy_classes_matrices=[]
            
            for conjclass in self._conjugacy_classes:
                sublist=[]
                #This line makes no sense, but conjclass.list()
                #returns an error
                clist=sg.ConjugacyClass(self._group,\
                conjclass.representative()).list()
                
                for element in clist:
                    sublist.append(np.array(element.matrix()))
                    
                self._conjugacy_classes_matrices.append(sublist)
            
        return list(self._conjugacy_classes_matrices)

    def get_numIRRs(self):
        '''
            返回 IRR 的数量
            
            IRR 是自同构群中的不可约表示。
        '''
        characters=self.get_character_table()
        numIRRs=len(characters[0])
        return numIRRs

    def get_IRR_degeneracies(self):
        '''
            返回一个表示每个 IRR 在置换矩阵表示中出现次数的列表
            
            该列表的顺序与字符表中的 IRR 顺序相同。
        '''
        if self._IRR_degeneracies==None:
            characters=self.get_character_table()
            numIRRs=len(characters[0])
            group_order=self.get_automorphism_group().order()
            
            self._IRR_degeneracies=[]
            matricies=self.get_conjugacy_classes_matrices()
            for i in range(numIRRs): #Loop over IRRs
                total=0
                for j in range(numIRRs): #loop over classes
                    total=total +float(len(matricies[j])/group_order)* np.conj(np.complex(characters[i][j]))* np.trace(matricies[j][0])
                
                self._IRR_degeneracies.append(round(np.real(total)))
                
        return list(self._IRR_degeneracies)
        
    def get_projection_operator(self, j):
        '''
            返回一个 numpy 数组，表示将矩阵投影到第 j 个 IRR 子空间的投影算子。
            
            参数：
            j - 字符表中的 IRR 的索引。
        '''
        degen=self.get_IRR_degeneracies()
        if self._projection_operators==None:
            self._projection_operators=[None]*len(degen)
        
        if self._projection_operators[j] == None:
            #the dimension of the IRR is the character of the identity.
            IRR_dimension=self.get_character_table()[j,0]
            group_order=self.get_automorphism_group().order()
            characters=self.get_character_table()[j]
            matricies=self.get_conjugacy_classes_matrices()
            result=np.zeros((self._nnodes,self._nnodes))
            
            for i in range(len(characters)):
                for mat in matricies[i]:
                    result=result+mat*complex(characters[i])
                    
            self._projection_operators[j]=result*np.float(IRR_dimension)/np.float(group_order)
        
        return self._projection_operators[j]

    def get_transformation_operator(self):
        '''
            返回转换到 IRR 坐标系的转换算子。
            
            该方法计算一个基于 IRR 的变换矩阵。
        '''
        epsilon=1e-12
        if self._transformation_operator==None:
            result=[]
            degens=self.get_IRR_degeneracies()
            total=0
            for j in range(len(degens)):
                IRR_dimension=self.get_character_table()[j,0]
                if degens[j]:
                    P=self.get_projection_operator(j)
                    U,W,VH = la.svd(P)
                    
                    
                    # Uncomment this output if you want to see it.
                    # print("Representation ", j, " with degeneracy ", degens[j])
                    # print("W=",W)
                    # print("dimension=",IRR_dimension)
                    
                    R1=int(IRR_dimension)*degens[j]
                    R2=0
                    total=total+R1 
                    for i,w in enumerate(W):
                        if np.abs(w-1.0)<epsilon:
                            R2=R2+1
                            result.append(VH[i])
                        
                    if R1!=R2:
                        print("Warning!")
                        print("Found, ",R2," singular vectors")
                        print("there should be", R1)
                        raise Exception
                        
                    #print("Rank=",R1,R2)
                    #print("P=",np.real(P))
                    
            # print("sum of dimension*degeneracy", total)

            self._transformation_operator=np.array(result)
        return self._transformation_operator.copy()
    