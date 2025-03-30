'''top3,intersection mix code'''
import scipy.io
import glob
import cv2
import numpy as np
import math
import statistics
import pandas as pd
import matplotlib.pyplot as plt

'''find model has predict label'''

def find_img_value(path,im_name):    
    label = []
    for i in im_name:            
        try:            
            mat = scipy.io.loadmat(path + '/' + i +'.mat')
            la = mat['class']
        except:
            la = np.array(())

        if la.size==0:
            label.append(0)
        else:   
            lab = la[0,0]
            label.append(lab)    
    nn = []
    find = 0
    for i,v in enumerate(label) :
        if v > find:
            nn.append(i)            
    cc = []        
    for c in nn:
        s = im_name[c]
        cc.append(s)
        
    return cc

'''delete outliers'''

def im_name_f(path,TC):
    # print(path)
    im_filename = glob.glob(path + '//*.mat')
    im_name = []

    for i in im_filename:
        n = i.split('/')[-1].split('.')[0]
        im_name.append(n)
    im_name.sort(key=int)
        
    #delete outlier
    if TC == True:
        for j in range(99,143):
            c = j
            c = str(c)
            im_name.remove(c)        
            im_name.sort(key=int)
    
    return im_name

'''morphological features'''

def mor(pre):
    
    ##object's pixels/pixel of full image
    x = 38/517
    y = 20/271
    '''CSA'''
    a = pre[:,:,0].astype(np.uint8)
    contours, hierarchy = cv2.findContours(a.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        
    cont = contours[0]   
    ar_pixel = cv2.contourArea(cont)
    ar = cv2.contourArea(cont)*x*y
    
    '''perimeter'''
    per = cv2.arcLength(cont,True)      
    
    '''circularity'''
    if per == 0:
        cir =0
    else:    
        cir = 4*math.pi*(ar_pixel/(per*per))
    
    '''centroid'''
    for c in contours:
	# calculate moments for each contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX,cY = 0, 0
            
    return ar,ar_pixel,per,cir,cX,cY

''' find the prediction to be calculated from multiple predictions'''

def top3_result(ma_path,inter):

    morpho = []
    d = []
    io = []
    label_ = []
    score_ = []
    b_idx = []
    
    for j in inter:        
        try:
            mat = scipy.io.loadmat(ma_path +'/'+ j +'.mat')
            ma = mat['mask']
            la = mat['class']
            sco = mat['score']
            m,n,z = ma.shape
        #incase of non-predict
        except:
            z=0
            
        gt = cv2.imread(gt_path + '/' + j +'.bmp').astype(np.float64)
        gt = gt[:,:,:]
        gt.astype(float)
        gt = (gt > 1)*1
     
        if z == 0:
            d.append(0)
            io.append(0)
            label_.append(0)
            score_.append(0)    
            morp = (0,0,0,0,0,0)
            morpho.append(morp)
            b_idx.append(None)
    
        if z == 1:
            pre = np.zeros((ma.shape[0],ma.shape[1],3))
            pre[:,:,:] = ma*255
            ##
            morp = mor(pre)
            morpho.append(morp)
            ##
            pre = (pre > 1)*1        
            a = 2*gt*pre
            b = gt + pre
            dice = (np.sum(a)+0.0000000001)/(np.sum(b)+0.0000000001)
            iou = (dice/(2-dice))
            d.append(dice)
            io.append(iou)
            
            lab = la[0,0]        
            scor = sco[0,0]
            label_.append(lab)
            score_.append(scor)
            b_idx.append(0)
            
        if z == 2:
            compare_d = []
            compare_io = []
            for i in range(0,2):
                pre = np.zeros((ma.shape[0],ma.shape[1],3))
                m = ma[:,:,i].astype(np.float64)*255
                pre[:,:,0] = m
                pre[:,:,1] = m
                pre[:,:,2] = m                  
                pre = (pre > 1)*1        
                a = 2*gt*pre
                b = gt + pre
                dice = (np.sum(a)+0.0000000001)/(np.sum(b)+0.0000000001)
                iou = (dice/(2-dice))
                compare_d.append(dice)
                compare_io.append(iou)
                
            idx = compare_d.index(max(compare_d))
            
            d.append(compare_d[idx])
            io.append(compare_io[idx])
            ##find correspond label & score
            lab = la[0,idx]        
            scor = sco[0,idx]
            label_.append(lab)
            score_.append(scor)
            b_idx.append(idx)
            ##find morph
            pre = np.zeros((ma.shape[0],ma.shape[1],3))
            m = ma[:,:,idx].astype(np.float64)*255
            pre[:,:,0] = m
            pre[:,:,1] = m
            pre[:,:,2] = m   
            morp = mor(pre)
            morpho.append(morp)
            
        if z>=3:
            compare_d = []
            compare_io = []
            for i in range(0,3):
                pre = np.zeros((ma.shape[0],ma.shape[1],3))
                m = ma[:,:,i].astype(np.float64)*255
                pre[:,:,0] = m
                pre[:,:,1] = m
                pre[:,:,2] = m        
                        
                pre = (pre > 1)*1        
                a = 2*gt*pre
                b = gt + pre
                dice = (np.sum(a)+0.0000000001)/(np.sum(b)+0.0000000001)
                iou = (dice/(2-dice))
                compare_d.append(dice)
                compare_io.append(iou)
                
            idx = compare_d.index(max(compare_d))
            
            d.append(compare_d[idx])
            io.append(compare_io[idx])    
            ##find correspond label & score
            lab = la[0,idx]        
            scor = sco[0,idx]            
            label_.append(lab)
            score_.append(scor)
            b_idx.append(idx)
            ##find morph
            pre = np.zeros((ma.shape[0],ma.shape[1],3))
            m = ma[:,:,idx].astype(np.float64)*255
            pre[:,:,0] = m
            pre[:,:,1] = m
            pre[:,:,2] = m   
            morp = mor(pre)
            morpho.append(morp)
            
        '''average dice,IOU'''        
    p = sum(d)/len(d)
    print(ma_path.split('/')[-1].split('.')[0],'為=')
    print()
    print('Dice=',p)
    std = statistics.stdev(d)
    print('Dice_std=',std)
    
    ai = sum(io)/len(io)
    print('IOU=',ai)
    stdi = statistics.stdev(io)
    print('IOU_std=',stdi)
    print()
        
    return d,io,morpho,label_,score_,b_idx

''' intersection of all models'''

def intersec_(L):
    
    inter = set(L[0]).intersection(*L[1:])
    inter = list(inter)
    inter.sort(key=int)
    return inter

def make_ans(inter):
    ans = []
    ans = np.zeros(len(inter))
    if str(473) in inter:
        p=inter.index(str(473))
        ans[0:p+1]=1
        ans[p+1:]=2
    else:
        bi = []
        for i in inter:
            if i <= str(473):
                bi.append(i)        
        p=inter.index(max(bi))       
        ans[0:p+1]=1
        ans[p+1:]=2
    return ans

'''ABOUT CALCULATE CLASSIFICATION'''

'''Positive,Negative'''

def cal(label,ans):
    
    TP = np.zeros(len(label))
    TN = np.zeros(len(label))
    FP = np.zeros(len(label))
    FN = np.zeros(len(label))
    
    per_tpr = []
    per_fpr = []
    
    for i in range(0,len(label)):
        #print(i)
        if  ans[i]==1 and  label[i]==1:
            TN[i]=1
        if  ans[i]==1 and  label[i]==2:
            FP[i]=1
        if  ans[i]==2 and  label[i]==2:
            TP[i]=1
        if  ans[i]==2 and  label[i]==1:
            FN[i]=1
    TP_ = sum(TP)
    TN_ = sum(TN)
    FP_ = sum(FP)
    FN_ = sum(FN)
    total_predict = TP_+TN_+FP_+FN_   
    Accuracy    = (TP_ + TN_)/len(label)   
    Sensitivity = TP_/(TP_+FN_)##Recall   
    Specificity = TN_/(FP_+TN_)  
    Precision   = TP_/(TP_+FP_)
    Fscore      = (2*Precision*Sensitivity)/(Precision+Sensitivity)
    total_acc = (TP_ + TN_)/total_predict    

    print(total_acc)
    print('Sensitivity=',Sensitivity)
    print('Specificity=',Specificity)
    print('Precision=',Precision)
    print('Fscore=',Fscore)
    print()
    print('TP=',int(TP_))
    print('TN=',int(TN_))
    print('FP=',int(FP_))
    print('FN=',int(FN_))
    print(int(total_predict))

    return total_acc,Sensitivity,Specificity,int(TP_),int(TN_),int(FP_),int(FN_),Accuracy,TP,TN,FP,FN,total_predict

'''(teacher) pulse non detect'''
def cal_p(label,ans):
    
    TP = np.zeros(len(label))
    TN = np.zeros(len(label))
    FP = np.zeros(len(label))
    FN = np.zeros(len(label))
    
    for i in range(0,len(label)):
        
        if  ans[i]==1 and  label[i]==1:
            TN[i]=1
        if  ans[i]==1 and  label[i]==2:
            FP[i]=1
        if  ans[i]==2 and  label[i]==2:
            TP[i]=1
        if  ans[i]==2 and  label[i]==1:
            FN[i]=1
        ##none detect ##    
        if  ans[i]==2 and  label[i]==0:
            FN[i]=1
        if  ans[i]==1 and  label[i]==0:
            TN[i]=1

    
    ##
    TP_ = sum(TP)
    TN_ = sum(TN)
    FP_ = sum(FP)
    FN_ = sum(FN)
    total_predict = TP_+TN_+FP_+FN_   
    Accuracy    = (TP_ + TN_)/len(label)   
    Sensitivity = TP_/(TP_+FN_)##Recall   
    Specificity = TN_/(FP_+TN_)  
    Precision   = TP_/(TP_+FP_)
    Fscore      = (2*Precision*Sensitivity)/(Precision+Sensitivity)
    total_acc = (TP_ + TN_)/total_predict    

    return total_acc,Sensitivity,Specificity,int(TP_),int(TN_),int(FP_),int(FN_),Accuracy,TP,TN,FP,FN

'''PQ'''
def PQ(label,TP,TN,FP,FN,d,iou,name,an):
    c_d = []
    c_i = []
    n_d = []
    n_i = []

    if type(label) == np.float64:
        l = int(label)        
    else:
        l = len(label)
    # print('PQ l=',l)
    for i in range((an == 1).sum(),l):
        
        ##positive
        if TP[i] == 0:
            c_PQ_d = 0
            c_PQ_i = 0
        else:
            c_PQ_d = (TP[i]/(TP[i]+(FP[i]*0.5)+(FN[i]*0.5)))*(d[i]/TP[i])  
            c_PQ_i = (TP[i]/(TP[i]+(FP[i]*0.5)+(FN[i]*0.5)))*(iou[i]/TP[i])
        c_d.append(c_PQ_d)
        c_i.append(c_PQ_i)
    
    for i in range((an == 1).sum()):
        
        ##negative
        if TN[i] == 0:
            PQ_d = 0
            PQ_i = 0
        else:
            PQ_d = (TN[i]/(TN[i]+(FN[i]*0.5)+(FP[i]*0.5)))*(d[i]/TN[i])  
            PQ_i = (TN[i]/(TN[i]+(FN[i]*0.5)+(FP[i]*0.5)))*(iou[i]/TN[i]) 
        n_d.append(PQ_d)
        n_i.append(PQ_i)
        
    if type(label) == np.float64:
        ##average pq
        aPQ_d = ((sum(c_d)/(an == 1).sum())+(sum(n_d)/(label-(an == 1).sum())))/2        
        aPQ_i = ((sum(c_i)/(label-(an == 1).sum()))+(sum(n_i)/(an == 1).sum()))/2
        
    else:    
        ##average pq
        aPQ_d = ((sum(c_d)/(an == 1).sum())+(sum(n_d)/(len(label)-(an == 1).sum())))/2
        aPQ_i = ((sum(c_i)/(len(label)-(an == 1).sum()))+(sum(n_i)/(an == 1).sum()))/2
        
    '''save to csv format'''
    save_PQ_d = []
    save_PQ_i = []
  
    save_PQ_d = c_d+n_d
    save_PQ_i = c_i+n_i

    ''''''   
    print()    
    stdi = statistics.stdev(save_PQ_i)

    print('aPQ_i=','%0.3f\u00B1%0.3f'%(aPQ_i,stdi))
    
    return c_d,c_i,n_d,n_i,save_PQ_d,save_PQ_i


'''ROC'''
from sklearn.metrics import roc_curve,auc
def skl_roc(p_label,p_score,NorC):
    '''
    

    Parameters
    ----------
    p_label : predicted class.
    p_score : predicted score.
    NorC : Normal or CTS, if Normal=1,if CTS=2.


    '''
    fpr,tpr,th = roc_curve(p_label,p_score,pos_label=NorC)
    auc_ = auc(fpr,tpr) 
    return fpr,tpr,auc_,th

def ROC_table(label,score_,mode):
    # class1 = []
    # class2 = []
    ##mode = 0: 0=none
    ##mode = 1: 0=normal
    
    p1     = []
    p2     = []
    if mode == 0:
        for i in range(0,len(label)):
            if label[i] == 1:
                p1.append(score_[i])
                p2.append(1-score_[i])
                
            if label[i] == 2:
                p1.append(1-score_[i])
                p2.append(score_[i])   
            if label[i] == 0:
                p1.append(0)
                p2.append(0) 
    if mode == 1:
        for i in range(0,len(label)):
            if label[i] == 0:
                p1.append(score_[i])
                p2.append(1-score_[i])
                
            if label[i] == 1:
                p1.append(score_[i])
                p2.append(1-score_[i])
                
            if label[i] == 2:
                p1.append(1-score_[i])
                p2.append(score_[i])   

    return p1,p2


colors = ['crimson',
          'orange',
          'gold',
          'mediumseagreen',
          'steelblue', 
          'mediumpurple',
          'violet',
          'purple',
          'dodgerblue',
          'royalblue',
          'navy',
          'darkblue',
          'mediumblue',
          'blue',
          'wheat',
          'tan',
          'chartreuse',
          'palegreen',
          'lightseagreen',
          'olivedrab',
          'darkkhaki'
          ]

def make_name(l):
    n = []
    for i in l:
        na = i.split('//')[-2]
        n.append(na)
    return n

'''chi-square'''
def chi_square(label):
    normal = 0
    positive    = 0
    for i in range(0,len(label)):
        if label[i] == 1:
            normal = normal + 1
            
        if label[i] == 2:
            positive = positive + 1
    print('normal',normal)
    print('positive',positive)
        
    return normal,positive

def vote(i,start,end,label):
    b1 = label[i-3]
    b2 = label[i-2]
    b3 = label[i-1]
    a1_ = i+1
    a2_ = i+2
    if a1_ >=len(label[start:end]):
        a1_ = a1_- len(label[start:end])

    if a2_ >=len(label[start:end]):
        a2_ = a2_- len(label[start:end])
    a1 = label[a1_]    
    a2 = label[a2_]
    
    t = (b1+b2+b3+a1+a2)/5

    if  t >= 1.5:
        p = 2
    if  t < 1.5:
        p = 1
    return p
                    

'''main code'''
def file_name(mode,TC,ROC,case_ROC):
    
    input_ = im_name_f(file_1,TC)
    
    '''find image index'''    
    if mode == 1:
        li = []
        input_image = input_
        
        print('Image Length=',len(input_image))
    
    if mode == 2:        
        li = []
        for i in ma_path:

            locals()['c%s' % (i)] = find_img_value(i, input_)            
            li.append(locals()['c%s' % (i)])
            
        input_image  = intersec_(li)
        print('Len of intersection image=',len(input_image))
        
    '''calculate segment and class result'''    
    '''make ans'''
    ans1 = make_ans(input_image)    
    
    ''' segment '''
    se_re = []
    
    '''class''' 
    cl_re = []
    clp_re = []
    # clm_re = []
    
    xy_1 = [] ##normal
    xy_2 = [] ##CTS
    
    r1 = []
    r2 = []
    
    '''chi'''
    chi = []
    
    '''case by case acc'''
    cacc = []
    cacal =[]
    '''PQ'''
    PQ_o = []
    PQ_n = []
    PQ_de = []
    
    for j in ma_path:
              
        locals()['S%s' % (j)] = top3_result(j, input_image)  
        ##class
        locals()['CL%s' % (j)] = cal(locals()['S%s' % (j)][3], ans1)
        locals()['ch%s' % (j)] = chi_square(locals()['S%s' % (j)][3])
        locals()['CLP%s' % (j)] = cal_p(locals()['S%s' % (j)][3], ans1)
        
        ##
        ##cal PQ
        ##636
        locals()['PQ%s' % (j)] = PQ(locals()['S%s' % (j)][3],locals()['CL%s' % (j)][8],locals()['CL%s' % (j)][9]
                                    ,locals()['CL%s' % (j)][10], locals()['CL%s' % (j)][11]
                                    ,locals()['S%s' % (j)][0],locals()['S%s' % (j)][1],'ori',ans1)

        PQ_o.append(locals()['PQ%s' % (j)])       
        se_re.append(locals()['S%s' % (j)])
        
        cl_re.append(locals()['CL%s' % (j)])
        clp_re.append(locals()['CLP%s' % (j)])        
        chi.append(locals()['ch%s' % (j)])   

    return input_image,li,se_re,cl_re,ans1,xy_1,xy_2,r1,r2,chi,clp_re,cacc,PQ_de

## save to csv
def make_csv(aa,ma_name,in1,in2,n,end):
    ori = pd.read_csv(csv_path)
    ren_d = ['slices_name']
    dice = [aa[0]]
    for i in range(0,len(ma_name)):
        if type(n) == str:
            re_d = n + str(len(aa[0])) + '_' + ma_name[i] + end
        else:    
            re_d = str(len(aa[0])) + '_' + ma_name[i] + end
        ren_d.append(re_d)   
        dice.append(aa[in1][i][in2])

    d_df = pd.DataFrame(dice)    
    d_df = d_df.T
    d_df.columns = ren_d    
    
    new_d = pd.concat([ori,d_df],axis = 1)
    new_d.to_csv(csv_path,index=None)
    
    return d_df,new_d

## save value from list(tuple[])
def t_make_csv(aa,ma_name,in1,in2,in3,n,end):
    
    ori = pd.read_csv(csv_path)
    ren_d = ['slices_name']
    
    dice = [aa[0]]
    
    for i in range(0,len(ma_name)):
        
        if type(n) == str:
            re_d = n + str(len(aa[0])) + '_' + ma_name[i] + end
        else:    
            re_d = str(len(aa[0])) + '_' + ma_name[i] + end
            
        ren_d.append(re_d)
        
        xy   = []      
        
        for j in range(0,len(aa[in1][i][in2])):
            
            xy.append(aa[in1][i][in2][j][in3])
            
        dice.append(xy)

    d_df = pd.DataFrame(dice)    
    d_df = d_df.T
    d_df.columns = ren_d    
    
    new_d = pd.concat([ori,d_df],axis = 1)
    new_d.to_csv(csv_path,index=None)
    return d_df,new_d
    
## make case centroid x & y csv
def c_make_csv(aa,ma_name,in1,in2,in3,c1,c2,n,end):
    
    ori = pd.read_csv(csv_path)
    ren_d = ['slices_name']
    
    dice = [aa[0][c1:c2]]
    
    for i in range(0,len(ma_name)):
        
        if type(n) == str:
            re_d = n + str(len(aa[0])) + '_' + ma_name[i] + end
        else:    
            re_d = str(len(aa[0])) + '_' + ma_name[i] + end
            
        ren_d.append(re_d)
        
        xy   = []      
        
        for j in range(c1,c2):
            
            xy.append(aa[in1][i][in2][j][in3])
            
        dice.append(xy)

    d_df = pd.DataFrame(dice)    
    d_df = d_df.T
    d_df.columns = ren_d    
    
    new_d = pd.concat([ori,d_df],axis = 1)
    new_d.to_csv(csv_path,index=None)
    return d_df,new_d
    
## find different##
def find_different(aa_,ma_name,mode):
    same_ = []
    diff = []
    if mode == 1:
        print('done')
    else:
        for i in range(0,len(ma_name)):        
            diff1 = []
            diff2 = []
            diff3 = []
            
            f1 = aa_[1][i]
            f1_name = ma_name[i]
            if i+1 == len(ma_name):            
                f2 = aa_[1][0]
                f2_name = ma_name[0]
            else:
                f2 = aa_[1][i+1]
                f2_name = ma_name[i+1]            
            f = [f1,f2]            
            same = intersec_(f)
            
            diff1 = set(f1)^set(same)
            diff2 = set(f2)^set(same)    
            diff3 = set(f1)^set(f2)
            
            print('****',f1_name,'&',f2_name,'****')
            print()
            print(f1_name,'>',len(diff1))
            print(diff1)
            print(f2_name,'>',len(diff2))        
            print(diff2)
            print(len(diff3))
            print()
            
            diff_set = [diff1,diff2]
            
            diff.append(diff_set)
            same_.append(same)
            
    return same_,diff


'''file name'''
gt_path = ('/gt')

file_1 = ''
file_2 = ''
file_3 = ''
file_4 = ''
file_5 = ''
file_6 = ''
file_7 = ''
file_8 = ''
file_9 = ''
file_10 = ''

'''main code detail'''

'''mode = 1 : 680slices(totall image) , mode = 2 : intersection '''
'''return 0: input_image = input image or intersection slices
          1: li = intersection detail(per model)
          2: se_re = [2][i]: d[0],io[1],morpho,label_[3],score_ 
          3: cl_re = total_acc,Sensitivity,Specificity,int(TP_),int(TN_),int(FP_),int(FN_),Accuracy 
          4: ans1 = ans
          15:non PQ'''
''''''

'''mode = 1 : normal , mode = 2 : CTS '''

ma_path = [file_1,file_2,file_3,file_4,file_5,
           file_6,file_7,file_8,file_9,file_10]

ma_name = ['Model1','Model2','Model3','Model4','Model5','Model6','Model7','Model8','Model9','Model10']

csv_path = r'.csv'
fig_path = r''        

mode = 1
# mode = 2
'''remember check whether delete outlier'''
aa = file_name(mode=mode,TC=True,ROC=True,case_ROC=True)

# d_csv = make_csv(aa, ma_name, in1=2, in2=0,n='dice',end='')

''' centroid fig'''

from matplotlib.pyplot import MultipleLocator

def inter_mean(li):
    
    for j in range(len(li)):
        if li[j]==0:
            i=j
            while i < len(li):
                if li[i]!=0:
                    j_1 = li[i]
                    # print('j_1=',j_1)
                    break
                i-=1
                
            i=j
            while i < len(li):
                if li[i]!=0:
                    j_2 = li[i]
                    # print('j_2=',j_2)
                    break
                i+=1
                if i == len(li):
                    i=0
            j_ = (j_1+j_2)/2
            # print('j_=',j_)
            li[j] = j_
    
    return li

def make_centroid_fig(aa,names,in1,in2,in3,c1,c2,fn):
    '''
    

    Parameters
    ----------
    in3 : 4=X,5=Y
    c1 : case start
    c2 : case end

    Returns
    -------
    cor : coordinate

    '''
    
    # cx = 3.8/517##38mm=3.8cm
    # cy = 2.0/271
    ##make coordinate ==> list[tuple(array,array)]
    cor = []    
    ##model
    for i in range(0,len(ma_name)):               
        lx = []        
        ##case
        for j in range(c1,c2):            
            lx.append(aa[in1][i][in2][j][in3])

        lx = inter_mean(lx)        
        
        arx = np.array(lx)
        print(names[i],'=',max(arx)-min(arx))
        ary = np.array(aa[0][c1:c2])            
        t = (ary,arx)
        
        cor.append(t)
        
    make_centroid_plt(names, cor, in3, c1, c2,fn)
        
    return cor

def make_centroid_plt(names,cor,in3,c1,c2,fn):
    
    plt.figure() 
    for (name,coor,col) in zip(names,cor,colors):
        
        # cplt = plt.gcf() 
        plt.plot(coor[0], coor[1], lw=1,color = col, label = name)  
              
        x_major_locator=MultipleLocator(5)       
        if in3 == 4:
            y_major_locator=MultipleLocator(25)        
        if in3 == 5:
            y_major_locator=MultipleLocator(5)
            
        ax=plt.gca()        
        ax.xaxis.set_major_locator(x_major_locator)        
        ax.yaxis.set_major_locator(y_major_locator)
        
            
        plt.xlabel('Slices')
        plt.ylabel('Pixel')
        
        plt.legend(bbox_to_anchor=(1,0),loc='lower left')

    if in3 ==4:
    # plt.savefig('Top3_%s%s'%(len(input_image),fig_name))  
        plt.savefig('Case%s-x'%(fn),bbox_inches='tight')
    if in3 ==5:
    # plt.savefig('Top3_%s%s'%(len(input_image),fig_name))  
        plt.savefig('Case%s-y'%(fn),bbox_inches='tight')
        
    plt.close() 
    # plt.show()
    return

case_li = [[0,47],[47,94],[94,138],[138,181],[181,225],[225,269],
            [269,316],[316,363],[363,410],[410,457],[457,504],[504,548],[548,592],[592,636]]


##delong test
import compare_auc_delong_xu

n_gt = aa[5][0]
c_gt = aa[5][1]

def delong_test(mode):
    
    if mode == 1:
        print('####delong test:normal####')
        lgt = n_gt
        m = aa[6]
    elif mode == 2:
        print('####delong test:CTS####')
        lgt = c_gt
        m = aa[7]
        
    for i in range(0,len(aa[6])):
        for j in range(i+1,len(aa[6])):
            m1 = m[i]
            m2 = m[j]
            print('m%sm%s=%0.3f'%(i,j,compare_auc_delong_xu.delong_roc_test(lgt,np.array(m1),np.array(m2))))
            
    return

# delong_test(mode=1)
# delong_test(mode=2)

##

def find_none_detect(aa_,number,ma_name):
    label_list = []
    
    find1 = []
    find2 = []
    find3 = []
    find4 = []
    
    none_list = []    
    
    for j in aa_[2]:
        label_list.append(j[3])##list all method's pridict label
    
    for i in range(0,len(label_list[0])): 
        #00101              
        if label_list[0][i] == 0:
           find1.append(number[i]) 
        none_list.append(find1) 
        
        if label_list[1][i] == 0:
           find2.append(number[i])
        none_list.append(find2)
        
        if label_list[2][i] == 0:
           find3.append(number[i])
        none_list.append(find3)
        
        if label_list[3][i] == 0:
           find4.append(number[i])      
        none_list.append(find4)
                   
    print()
    for i in range(0,len(ma_name)):
            
        print(ma_name[i],'共%s張='%len(none_list[i]),none_list[i])
        
    return none_list

# bb = find_none_detect(aa,aa[0], ma_name)
                
            
'''make boundary'''
from skimage.io import imread , imshow , imsave

def edge(path,i,r,g,b,mode,l_n):
    
    ##gt = 0
    if mode == 0:
        img = imread(path + '//' + str(i) + '.bmp')/255
    ##pre = 1
    if mode == 1:        
        nn = aa[0].index(str(i))
        n  = aa[2][l_n][5][nn]
        mat = scipy.io.loadmat(path + '/' + str(i) +'.mat')
        img = mat['mask'][:,:,n]
        
    edge = np.zeros((img.shape[0],img.shape[1],3))
   
    for c in range(1,img.shape[0]-1):
        for k in range(1,img.shape[1]-1):
            
            if img[c,k] == 1:
                cc = img[c-1:c+2,k-1:k+2]
                bb = np.sum(cc)
                bb = int(bb)
                if bb < 9:
                    edge[c,k,0] = r
                    edge[c,k,1] = g
                    edge[c,k,2] = b                    
                elif bb == 9:
                    edge[c,k,0] = 0
                    edge[c,k,1] = 0
                    edge[c,k,2] = 0
                    if c-1 == 0:                        
                        edge[c,k,0] = r
                        edge[c-1,k-1:k+1,1] = g
                        edge[c,k,2] = b                  
                    elif c+1 == 512:
                        edge[c,k,0] = r
                        edge[c+1,k-1:k+1,1] = g
                        edge[c,k,2] = b
                    elif k-1 == 0:
                        edge[c,k,0] = r
                        edge[c,k-1,1] = g
                        edge[c,k,2] = b
                    elif k+1 == 512:
                        edge[c,k,0] = r
                        edge[c,k+1,1] = g
                        edge[c,k,2] = b       
    return edge

def ori(path,i):
    ori = imread(path + '//' + str(i) + '.bmp')
    return ori


def merge(path,i,r,g,b,m):
    img = imread(path + '//' + str(i) + '.bmp')    
    color_p = np.zeros((img.shape[0],img.shape[1],3))
   
    for c in range(1,img.shape[0]-1):
        for k in range(1,img.shape[1]-1):
            if img[c,k]>0:
               color_p[c,k,0] = r
               color_p[c,k,1] = g
               color_p[c,k,2] = b
            
            
    return color_p*m

def only_b(img_path,gt_path,file_path,n,m,l_n):
    
    ori_img = ori(img_path,n)
    
    a = edge(gt_path,n,r=255,g=0,b=0,mode=0,l_n=l_n)##gt is red
    # a_ = merge(gt_path,n,r=255,g=0,b=0,m=m)
    
    p = edge(file_path,n,r=0,g=255,b=0,mode=1,l_n=l_n)##predict is green
    # p_ = merge(file_path,n,r=0,g=255,b=0,m=m)
    if p.shape == (0,0,3):
        p = np.zeros((a.shape[0],a.shape[1],3))
    overlap = a+p
    # overlap_ = a_+p_
    
    ##boundary overlap = yellow
    for l in range(3):
        for i in range(overlap.shape[0]):
            for j in range(overlap.shape[1]):
                if overlap[i,j,l] > 255:
                    overlap[i,j,0] =255
                    overlap[i,j,1] =255
                    overlap[i,j,2] =0
                    
    com = ori_img + overlap
    
    ##make sure boundary == yellow
    for l in range(3):
        for i in range(com.shape[0]):
            for j in range(com.shape[1]):
                if overlap[i,j,l] > 0:
                    com[i,j,l] = overlap[i,j,l]
    return com
import os

img_path = (r'')#original
IP1_path = (r'')##image processing
IP2_path = (r'')##image processing
IP3_path = (r'')##image processing

o_p = [img_path,img_path,IP1_path,IP2_path,IP3_path]
save_path = r''

def result_boundary(n,m):
    os.makedirs(save_path + '//' + str(n))
    j = 0
    for i in ma_path:        
        l_n = ma_path.index(i)
        p1 = only_b(o_p[j],gt_path,i,n=n,m=m,l_n=l_n)
        j = j+1
        name = i.split('//')[-1].split('_')[0]
        
        imsave(save_path +'//' + str(n) + '//' + name + '.bmp',p1.astype(np.uint8))    
    
    return 

# for i in range(len(bb)):
#     save_path = r'C:\Users\Tippy\Desktop\overlap\after'+'//'+str(o_pa[i])
#     for j in bb[i]:        
#         result_boundary(j,0.3)


def find_better(aa,ans,number):
    label_list = []
    
    find1 = []
      
    for j in aa[2]:
        label_list.append(j[3])
    # label_list = np.array(label_list)    
    for i in range(0,len(label_list[0])): 
        
        if (label_list[0][i] != int(ans[i]) and
            label_list[1][i] == int(ans[i]) and
            label_list[2][i] == int(ans[i])):
            find1.append(number[i])

    
    print()
    #00101
    print('m2x m34o共%s張='%len(find1),find1)

    return find1
     
bb = find_better(aa, aa[4],aa[0])  

