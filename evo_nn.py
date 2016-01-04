#population=[]
#fitness=[]

"""""""""""""""""""""""""""""""""""""""""""""""""""
Evo_NN:
___________
Evolutionary algorithm with pybrain Neural net used as fitness evaluating function. NN is trained on encoded songbits 
as well as random noise of equal length. Generations are refined until signal is heard.


From this point (270900 neurons 59 song data set) i can improve output songs via several methods:

1. Use songs from single artist and test output songs according to how many songs used in 
    the data set. ie (10, 20, 30...60)

3. Develope algorithm for directional, distructive and moderating selection. Make a dynamic fitness_evaluater
    that can be trained along disired path then continue unguided.

"""""""""""""""""""""""""""""""""""""""""""""""""""

from scipy.io.wavfile import write
from pybrain.structure import *
import numpy as np 
import random
from random import randint
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from scipy import fft, ifft
import numpy
from numpy import int16
import copy
from itertools import chain, izip
import math
from bregman.suite import *
from pyevolve import G1DList, GSimpleGA
from pyevolve import Mutators, Crossovers
#z=list(chain.from_iterable(izip(f,s)))
#k.real=z[1::2]
#k.imag=z[::2]
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ones=1
threes=3

l=5*44100
l=int(l)
ds=SupervisedDataSet(l,1)

def Genetic_algorithm(example):
    genome=G1DList.G1DList(len(example))
    genome.evaluator.set(eval_func)
    genome.setParams(rangemin=min(k), rangemax=max(k))
    genome.mutator.set(Mutators.G1DListMutatorRealGaussian)
    
    ga=GSimpleGA.GSimpleGA(genome)
    ga.setGenerations(34)
    ga.setPopulationSize(60)
    ga.terminationCriteria.set(ConvergenceCriteria)
    return ga

def pop_maker(ga_pop):
    ar=[]
    for elem in ga_pop:
        new=[]
        for i in elem:
            new.append(i)
        ar.append(new)
    count=0
    while count<len(ar):
        if ar[count]==[]:
            del ar[count]
        count+=1
    return ar


def populizer(l):
    fs=44100.0
    r3=randint(10,880)
    wav_a=[]
    co=0
    dt=1/fs 
    pop=[]
    coun=0
    for i in range(0,int(l)):
        wav_a.append(co)
        co=co+dt
    for freq in range(50,850):
        coun+=1
        amp=randint(200,20000)
        child=copy.deepcopy(wav_a)
        if freq%10==0:    
            for i in range(0,len(wav_a)):
                child[i]=math.sin(wav_a[i]*2*math.pi*freq)*amp
            pop.append(child)
    print('populizer populated')
    return pop

def eval_func(chromosome):
    in_arr=[]
    in_arr.append(chromosome[:])
    score=net.activate(in_arr[0])
    score=abs(score)
    score=.1/score[0]
    return score

def ConvergenceCriteria(ga):
    count=0
    error=.005
    pop=ga.getPopulation()
    for elem in pop:
        if eval_func(elem)>=200:
            count+=1
    if count>len(pop)/2:
        return True 
    return False



def sfft(signal):
    mag_array=[]
    x=LinearFrequencySpectrum(signal,nfft=1024,wfft=1024,nhop=512)
    for elem in x.STFT:
        for i in elem:
            mag_array.append(((i.real)*2)/1024.0)
    return mag_array

def CQFT(signal):
    p = default_feature_params()
    p['nhop'] = 256 #FFT hop size
    p['nfft']=1024  #FFT length for filterbank
    p['wfft']=512   #FFT signal window length
    p['feature']='cqft'
    F = Features(signal,p)
    out_array=[]
    for elem in F.X:
        for i in elem:
            out_array.append(i)
    return out_array

def sig_Chroma(signal):
    p = default_feature_params()
    p['nhop'] = 256
    p['nfft']=1024
    p['wfft']=512
    p['feature']='chroma'
    F=Features(signal,p)
    out_array=[]
    for elem in F.X:
        for i in elem:
            out_array.append(i)
    return out_array

def Mel_freq_cepstral_coef(signal):
    p=default_feature_params()
    p['nhop'] = 512
    p['wfft']=2048
    p['nfft']=2048
    p['feature']='mfcc'
    F=Features(signal,p)
    out_array=[]
    for elem in F.X:
        for i in elem:
            out_array.append(i)
    return out_array
    
def self_similarity(signal,feature):
    out_ar=[]
    '''described as Self-similiarity matrix 
       of signal using CQFT or CHROMA features'''
    feat=feature(signal)
    '''D here is the normed Euclidean distance between two matrices'''
    D=distance.euc_normed(feat.T,feat.T)
    for elem in D:
        for i in elem:
            out_ar.append(i)
    return out_ar

def similarity(array1,array2):
    count=0.0
    for i in range(1,len(array1)-1):
        if array1[i]==array2[i]:
            count+=1.0
    percent_similiar=count/len(array1)
    return percent_similiar*100

def self_similarity_shingles(signal, feature):
    '''Self similarity of signal using
       overlapping stacked CQFT or CHROMA features'''
    out_array=[]
    feat=feature(signal)
    X=adb.stack_vectors(feat.T,5,1)
    '''
    creates an overlapping stacked vector sequence from a series of vectors
    
    data - row-wise multidimensional data to stack 
    win - number of consecutive vectors to stack [1] 
    hop - number of vectors to advance per stack [1] 
    '''   
    D=distance.euc_normed(X,X)
    for elem in D:
        for i in elem:
            out_array.append(i)
    return out_array 


sig_proc_list=[CQFT,sfft,Mel_freq_cepstral_coef,sig_Chroma]

''' DYNAMIC TIME WARPING: USE ON CHROMA
p,q,D = dtw(M) 
Use dynamic programming to find a min-cost path through matrix M.
Return state sequence in p,q, and cost matrix in D     
'''


'''takes in new optimized populizations and pairs 
   it with step below whats considered perfect'''
def update_ds(new_data):
    for elem in ds:
        if elem[1][0] !=0:
            elem[1][0]+=1.5
    for elem in new_data:
        ds.addSample((elem),(threes))

def update_ds_keep_pop(new_data, ds, net):
    ds_new=SupervisedDataSet(l,1)
    for elem in ds:
        if elem[1][0] ==0:
            ds_new.addSample(elem[0],elem[1])
    for elem in new_data:
        ds_new.addSample(elem,(net.activate(elem)+3.0))
    return ds_new

def mutate(array1):
    """rand_num=randint(0,2)
    if rand_num%2==0:"""
    return mutate_rearange(array1)
    """
    if rand_num==1:
        return sin_wave_insert(array1)
    """

def sin_wave_insert(array1):
    r=randint(100,10000)
    swell=0
    r3=randint(10,680)
    '''add sin wave'''
    amp=randint(20,20000)
    fs=44100.0
    wav_a=[]
    co=0
    dt=1/fs 
    if 2%2==0:
        for i in range(0,int(l)):
            wav_a.append(co)
            co=co+dt
        for i in range(0,len(wav_a)):
            wav_a[i]=math.sin(wav_a[i]*2*math.pi*r3)*amp
        a=randint(0,len(array1)-1)
        b=randint(0,len(array1)-1)
        if b<a:
            c=b
            b=a
            a=c
        count=0
        if (b-a)>len(array1)/2.0:
            if randint(0,1)==1:
                b=b-((b-a)/2)
            else:
                a=a+((b-a)/2)
        for ii in range(a,b):
            array1[ii]=array1[ii]+wav_a[count]
            count+=1
        '''add timbre of some random instrument'''
    return array1

def mutate_rearange(array1):
    #choose two random places
    r=randint(100,10000)
    swell=0
    while swell<r:
        a=randint(0,len(array1)-1)
        b=randint(0,len(array1)-1)
        #swaps two geens over, could do other mutations
        c=array1[a]
        array1[a]=array1[b]
        array1[b]=c
        swell+=1
    return array1

def hybrid_reproduction(array1,array2,percentage):
    baby=[]
    it=0
    start=randint(0,len(array1)-1)
    stop=randint(0,len(array2)-1)
    if start>stop:
        stop, start=start,stop
    choice=randint(0,1)
    if choice==0:
        choosen=array1
    if choice==1:
        choosen=array2
    while it<(stop-start):
        choosen[start+it]=(array1[start+it]+array2[start+it])
        it+=1
    if randint(0,100)>percentage:
        choosen=mutate(choosen)
    return choosen

def reproduce_input(array1,array2,percentage):
    case_num=randint(0,2)
    if case_num%2==0:
        return one_point_recombine(array1,array2,percentage)
    if case_num==1:
        return hybrid_reproduction(array1,array2,percentage)

def one_point_recombine(array1,array2,percentage):
    #first Baby (mother first then father)
    #could mix more randomly
     #some element might get missed or overlapped
    baby=copy.deepcopy(array1)
    haploid=copy.deepcopy(array2)
    start=randint(0,len(haploid)-1)
    stop=randint(0,len(array2)-1)
    if start>stop:
        stop, start=start,stop
    baby[start:stop]=haploid[start:stop]
    #add a little variation by giving a percentage chance of mutation
    if randint(0,100)>percentage:
        baby=mutate(baby)
    return baby


def brain_games(l):
    net=buildNetwork(l,1000,30,1)
    print('net created')
    return net

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
MAIN: First runs maino to evolve noise through NN operating on trends in pcm

Then writes all members of population to desktop. Next writes old population 
into program and selects population with NN based on trend of alternative aspect
of pcm data through signal processing functions.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''





'''Creates net then runs evo algorithm till best population is reached. At this point it 
updates the dataset and retrains the network then running the evo algorithm again for increased
quality.'''


test_array=[]
def maino(iters):
    net=brain_games()
    trainer=BackpropTrainer(net,ds)
    trains=0
    while trains<4:
        f=trainer.train()
        print('net error '+str(f))
        trains+=1
    count=0
    while count<iters:
        pop=populize(k)
        top_3=generation(pop,net)
        update_ds(top_3)
        write('lvl_'+str(count)+'.wav',44100,int16(top_3[0]))
        trainer=None
        net, trainer=train_till(net,ds,5)
        count+=1
        print('ON '+str(count))
    return top_3

def make_trainnet(ds,f):
    net=brain_games(f)
    trainer=BackpropTrainer(net,ds)
    trains=0
    while trains<5:
        f=trainer.train()
        print('net error '+str(f))
        trains+=1
    return net, trainer

def main_i(net,operator,populi):
    fullpop=[]
    count=0
    while count<1:
        pop_new=generation(populi,net,operator)
        print('recieved '+str(len(pop_new)))
        for elem in pop_new:
            fullpop.append(elem)
        print('have '+str(len(fullpop))+' members in pop')
        #populi=populizer(l)
        if len(fullpop)>=50:    
            count+=1
    return fullpop



def main_keep_pop(iters,ds,pop,operator):
    net=brain_games()
    trainer=BackpropTrainer(net,ds)
    top=3
    trains=0
    while trains<4:
        f=trainer.train()
        print('net error '+str(f))
        trains+=1
    count=0
    while count<iters:
        pop_new=generation(pop,net,operator)
        #ds=update_ds_keep_pop(pop_new[0:3],ds, net)
        trainer=None
        #net, trainer=train_till(net,ds,5)
        print('ON '+str(count))
        count+=1
    write_tst_songs(pop_new,str(operator))
    return pop_new

'''Writes one song from each training group of an array of an array of generated songs'''
def write_tst_songs(full_all,operator):
    count=0
    for elem in full_all:
        write('Pop_o_'+operator+str(count)+'.wav',44100,int16(elem))
        count+=1
    return 0

def populize_from_top(ranger,stringV):
    pop=[]
    for i in range(0,ranger):
        x=np.fromfile(open('Pop_o_'+stringV+str(i)+'.wav'),np.int16)
        pop.append(x)
    return pop 
    ""

'''takes Net and Data set and trains network until it is proven to be accurate
    to within a specified error. Accuracy determined by testing net on tstdata
    and evaluating the mean margin of error'''
def train_till(net, ds,iter_bug):
    tt=False
    improvement=.3
    trainer=BackpropTrainer(net,ds)
    old=trainer.train()
    print('first trainer error '+str(old))
    count=0
    while tt!=True:
        new=trainer.train()
        print('trainer error ' +str(new))
        grade=old-new
        if grade<improvement or count>=iter_bug:
            tt=True
        count+=1
        old=new
    return net, trainer 


def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

def delete_same(population,fitness):
    keeper=[]
    keeperpop=[]
    total=0
    count=0
    while count<len(fitness):
        locs=list_duplicates_of(fitness, fitness[count])
        if len(locs)>1:
            total+=len(locs)-1
            for elem in locs:
                print('deleting '+str(fitness[elem]))
            keeper.append(fitness[locs[0]])
            keeperpop.append(population[locs[0]])
            fitness = [i for j, i in enumerate(fitness) if j not in locs]
            population = [i for j, i in enumerate(population) if j not in locs]
        count+=1
    for i in range(0,len(keeper)):
        fitness.append(keeper[i])
        population.append(keeperpop[i])
    return population, fitness, total

def generation(population,fitness_eval, operator):
    cont=0
    fitness=[3]
    return_pop=[]
    death_per=.36
    error=.009
    popsize=len(population)
    kill_limit=death_per*popsize
    k=0
    print('in generation')
    Condition=False
    while Condition==False:
        fitness=[]
        if operator!=None:
            for i in range(0,popsize):
                fitness.append(abs(fitness_eval.activate(operator(population[i]))))
        if operator==None:
            if k==0:
                print('in operator==None')
            for i in range(0,popsize):
                fitness.append(abs(fitness_eval.activate(population[i])))
                
        if k==0:
            baseline=numpy.std(fitness)
            k+=1
        killed=0
        #starting at lowest fitness then increasing untill kill limit reached
        population, fitness, killed=delete_same(population, fitness)
        while killed<kill_limit:
            ii=0
            while ii<len(population):
                maxx=max(fitness)
                if fitness[ii]==maxx:
                    #this bit removes bad gene from population and from fitness
                    #rint('killed '+str(fitness[ii]))
                    population.pop(ii)
                    fitness.pop(ii)
                    killed+=1
                if killed==kill_limit:
                    break
                ii+=1
        babies=0
        cpop=len(population)-1 #current population
        if cont%10==0:
            print('best member '+ str(min(fitness)))
            for elem in fitness:
                print('population fitness:')
                print(str(elem))
        passed=0
        for elem in fitness:
            if elem<error:
                passed+=1
            if passed>15:
                Condition=True
                break
                
        print('passed '+str(passed))
        if passed>12:
            break
        copy_pop=copy.deepcopy(population)
        #std_dev=numpy.std(fitness)
        percentagee=55  
        #print('percentage '+ str(percentagee))
        while babies<killed:
            #produces two babies from two random parents
            o=randint(0,cpop)
            oo=randint(0,cpop)
            while o==oo:
                oo=randint(0,cpop)
            population.append(reproduce_input(copy_pop[o],copy_pop[oo],percentagee))
            babies+=1
        cont+=1
        print('generation   '+str(cont))
        if cont==500:
            break
    fit_bank=[]    
    for i in range(0,len(fitness)-1):
        if fitness[i]<error:
            if fitness[i] not in fit_bank:
                return_pop.append(population[i])
                fit_bank.append(fitness[i])
    return return_pop

""""""""""""""""""""""""""""""""""""""""""""""""""""
List pcm of songs and add them to array song_array 
with target

Then iterate through array and applay selected operator

Create NN with input len of output array from operator
and ~~Rule of Thumb~~ for number of nodes in build net framework

"""""""""""""""""""""""""""""""""""""""""""""""""""

def find_trend_of(song_array,operator):
    count=0
    for elem in song_array:
        input_size=len(operator(elem[0]))
    print('this is input size '+str(input_size))
    ds=SupervisedDataSet(input_size,1)
    for elem in song_array:
        val=operator(elem[0])
        print('lenght of sample '+str(len(val)))
        ds.addSample(val,elem[1])
    print('ds complete')
    return ds


def analyze_accuracy(song_array,operator,net):
    count=0
    err_stats=[]
    for elem in song_array:
        netval=net.activate(elem[0])
        print('net got '+str(netval)+' while theortical value is '+str(elem[1]))
        centerr=(abs(netval-elem[1]))/float(elem[1])*100
        err_stats.append(abs(netval-elem[1]))
        print('percent error '+str(centerr))
    print('this is median '+ str(np.median(err_stats)))
    return 0

populus=populizer(l)

x=np.fromfile(open('Crooked_Smile.wav'),np.int16)
k=x[0*88200:(0*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('born_to_run.wav'),np.int16)
k=x[4*88200:(4*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
ds.addSample(k,(0))

x=populus[2]
x=int16(x)
ds.addSample(x,(2.5))



x=populus[48]
x=int16(x)
ds.addSample(x,(2.5))

x=np.fromfile(open('Tequila_Remix.wav'),np.int16)
k=x[25*88200:(25*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=populus[35]
x=int16(x)
ds.addSample(x,(2.5))

x=np.fromfile(open('Story_of_My_Life.wav'),np.int16)
k=x[84*88200:(84*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=populus[29]
x=int16(x)
ds.addSample(x,(2.5))

x=np.fromfile(open('Can_I_Get_Witcha.wav'),np.int16)
k=x[0*82200:(0*82200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Mo_money.wav'),np.int16)
k=x[2*88200:(2*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=populus[9]
x=int16(x)
ds.addSample(x,(2.5))

x=np.fromfile(open('Stay_Awake.wav'),np.int16)
k=x[23*88200:(23*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=populus[43]
x=int16(x)
ds.addSample(x,(2))


x=numpy.random.rand(l)
x=x*1000
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(x,(3))

x=np.fromfile(open('you_make_me.wav'),np.int16)
k=x[85*88200:(85*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=populus[29]
x=int16(x)
ds.addSample(x,(2.5))

x=np.fromfile(open('all_you_need.wav'),np.int16)
k=x[129*88200:(129*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Wake_me_up.wav'),np.int16)
k=x[84*88200:(84*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Trouble_on_mind.wav'),np.int16)
k=x[1*88200:(1*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Jungle.wav'),np.int16)
k=x[5*88200:(5*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('show_goes.wav'),np.int16)
k=x[82*88200:(82*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Baba.wav'),np.int16)
k=x[58*88200:(58*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Tennis_Court.wav'),np.int16)
k=x[43*88200:(43*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Snow.wav'),np.int16)
k=x[18*88200:(18*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('I_got_soul.wav'),np.int16)
k=x[187*88200:(187*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Its_Alright.wav'),np.int16)
k=x[18*88200:(18*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('watchtower.wav'),np.int16)
k=x[9*88200:(9*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('blessed.wav'),np.int16)
k=x[106*88200:(106*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('good_fun.wav'),np.int16)
k=x[73*88200:(73*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('This_One.wav'),np.int16)
k=x[73*88200:(73*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=numpy.random.rand(l)
x=x*500
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
x=int16(x)
ds.addSample(x,(3))

x=np.fromfile(open('Goldie.wav'),np.int16)
k=x[73*88200:(73*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=numpy.random.rand(l)
x=x*500
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
x=int16(x)
ds.addSample(x,(3))

x=numpy.random.rand(l)
x=int16(x)
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(x,(3))

x=numpy.random.rand(l)
x=x*30000
x=int16(x)
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(x,(3))

x=numpy.random.rand(l)
x=x*1000
x=int16(x)
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(x,(3))

x=numpy.random.rand(l)
x=x*31300
x=int16(x)
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(x,(3))

x=np.fromfile(open('By_The_Way.wav'),np.int16)
k=x[164*88200:(164*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('years.wav'),np.int16)
k=x[124*88200:(124*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=numpy.random.rand(l)
x=x*500
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
x=int16(x)
ds.addSample(x,(3))

x=np.fromfile(open('we_run.wav'),np.int16)
k=x[78*88200:(78*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('here_comes_sun.wav'),np.int16)
k=x[15*88200:(15*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=numpy.random.rand(l)
x=x*500
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
x=int16(x)
ds.addSample(x,(3))

x=np.fromfile(open('All_of_the_Lights.wav'),np.int16)
k=x[8*88200:(8*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Animals.wav'),np.int16)
k=x[82*88200:(82*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=numpy.random.rand(l)
x=x*500
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
x=int16(x)
ds.addSample(x,(3))

x=np.fromfile(open('higher_ground.wav'),np.int16)
k=x[47*88200:(47*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('lost_at_sea.wav'),np.int16)
k=x[45*88200:(45*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))

ds.addSample(k,(0))

x=np.fromfile(open('Changes.wav'),np.int16)
k=x[60*88200:(60*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=numpy.random.rand(l)
x=x*500
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
x=int16(x)
ds.addSample(x,(3))

x=np.fromfile(open('How Do U Want It.wav'),np.int16)
k=x[209*88200:(209*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Xoxo.wav'),np.int16)
k=x[46*88200:(46*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Band_on_the_run.wav'),np.int16)
k=x[130*88200:(130*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=numpy.random.rand(l)
x=x*1000
x=int16(x)
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(x,(3))

x=numpy.random.rand(l)
x=x*1000
x=int16(x)
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(x,(3))

x=numpy.random.rand(l)
x=x*1000
x=int16(x)
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(x,(3))

x=populus[23]
x=int16(x)
ds.addSample(x,(2))

x=numpy.random.rand(l)
x=x*1000
x=int16(x)
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(x,(3))

x=np.fromfile(open('Dear_Body.wav'),np.int16)
k=x[60*88200:(60*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=populus[5]
x=int16(x)
ds.addSample(x,(2))

x=numpy.random.rand(l)
x=x*500
x=int16(x)
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(x,(3))

x=np.fromfile(open('Molly.wav'),np.int16)
k=x[38*88200:(38*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=numpy.random.rand(l)
x=x*1000
x=int16(x)
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(x,(3))

x=np.fromfile(open('House_Of_Gold.wav'),np.int16)
k=x[78*88200:(78*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('fake_you_out.wav'),np.int16)
k=x[68*88200:(68*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Highway_to_Hell.wav'),np.int16)
k=x[52*88200:(52*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('journey.wav'),np.int16)
k=x[82*88200:(82*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=numpy.random.rand(l)
x=x*1000
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(x,(3))

x=np.fromfile(open('Operation Ground and Pound'),np.int16)
k=x[0*88200:(0*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=populus[33]
x=int16(x)
ds.addSample(x,(2))

x=numpy.random.rand(l)
x=x*1000
x=int16(x)
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(3))

x=numpy.random.rand(l)
x=x*1000
x=int16(x)
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(x,(3))



x=np.fromfile(open('Dear_Body.wav'),np.int16)
k=x[43*88200:(43*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Spectrum.wav'),np.int16)
k=x[32*88200:(32*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))





'''
x=np.fromfile(open('the_reaper.wav'),np.int16)
k=x[51*82200:(51*82200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('jubel.wav'),np.int16)
k=x[146*82200:(146*82200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('IZ.wav'),np.int16)
k=x[17*82200:(17*82200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Crooked_Smile.wav'),np.int16)
k=x[0*88200:(0*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Tequila_Remix.wav'),np.int16)
k=x[25*88200:(25*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Story_of_My_Life.wav'),np.int16)
k=x[84*88200:(84*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Can_I_Get_Witcha.wav'),np.int16)
k=x[0*82200:(0*82200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Mo_money.wav'),np.int16)
k=x[2*88200:(2*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Stay_Awake.wav'),np.int16)
k=x[23*88200:(23*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=numpy.random.rand(l)
x=x*1000
k=pcm_to_fft(x,0,513)
k=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(3))

x=np.fromfile(open('you_make_me.wav'),np.int16)
k=x[85*88200:(85*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('all_you_need.wav'),np.int16)
k=x[129*88200:(129*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Wake_me_up.wav'),np.int16)
k=x[84*88200:(84*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Trouble_on_mind.wav'),np.int16)
k=x[1*88200:(1*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Jungle.wav'),np.int16)
k=x[5*88200:(5*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('show_goes.wav'),np.int16)
k=x[82*88200:(82*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Baba.wav'),np.int16)
k=x[58*88200:(58*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Tennis_Court.wav'),np.int16)
k=x[43*88200:(43*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Snow.wav'),np.int16)
k=x[18*88200:(18*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('I_got_soul.wav'),np.int16)
k=x[187*88200:(187*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Its_Alright.wav'),np.int16)
k=x[18*88200:(18*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
#k=pcm_to_fft(k,0,450)
k=pcm_to_fft(k,0,513)
k=k.real
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('watchtower.wav'),np.int16)
k=x[9*88200:(9*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('blessed.wav'),np.int16)
k=x[106*88200:(106*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('good_fun.wav'),np.int16)
k=x[73*88200:(73*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('This_One.wav'),np.int16)
k=x[73*88200:(73*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(1))


x=np.fromfile(open('Goldie.wav'),np.int16)
k=x[73*88200:(73*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=numpy.random.rand(l)
x=x*500
k=pcm_to_fft(x,0,513)
k=k.real
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(3))

x=numpy.random.rand(l)
k=pcm_to_fft(x,0,513)
k=k.real
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(3))

x=numpy.random.rand(l)
x=x*30000
k=pcm_to_fft(x,0,513)
k=k.real
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(3))

x=numpy.random.rand(l)
x=x*1000
k=pcm_to_fft(x,0,513)
k=k.real
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(3))

x=numpy.random.rand(l)
x=x*31300
k=pcm_to_fft(x,0,513)
k=k.real
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(3))

x=np.fromfile(open('By_The_Way.wav'),np.int16)
k=x[164*88200:(164*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('years.wav'),np.int16)
k=x[124*88200:(124*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('we_run.wav'),np.int16)
k=x[78*88200:(78*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('here_comes_sun.wav'),np.int16)
k=x[15*88200:(15*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('All_of_the_Lights.wav'),np.int16)
k=x[8*88200:(8*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Animals.wav'),np.int16)
k=x[82*88200:(82*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('higher_ground.wav'),np.int16)
k=x[47*88200:(47*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('lost_at_sea.wav'),np.int16)
k=x[45*88200:(45*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real

ds.addSample(k,(0))

x=np.fromfile(open('Changes.wav'),np.int16)
k=x[60*88200:(60*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('How Do U Want It.wav'),np.int16)
k=x[209*88200:(209*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Xoxo.wav'),np.int16)
k=x[46*88200:(46*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Band_on_the_run.wav'),np.int16)
k=x[130*88200:(130*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=numpy.random.rand(l)
x=x*1000
k=pcm_to_fft(x,0,513)
k=k.real
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(3))

x=numpy.random.rand(l)
x=x*1000
k=pcm_to_fft(x,0,513)
k=k.real
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(3))

x=numpy.random.rand(l)
x=x*1000
k=pcm_to_fft(x,0,513)
k=k.real
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(3))

x=numpy.random.rand(l)
x=x*1000
k=pcm_to_fft(x,0,513)
k=k.real
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(3))

x=numpy.random.rand(l)
x=x*1000
k=pcm_to_fft(x,0,513)
k=k.real
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(3))

x=np.fromfile(open('Dear_Body.wav'),np.int16)
k=x[60*88200:(60*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Spectrum.wav'),np.int16)
k=x[32*88200:(32*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=numpy.random.rand(l)
x=x*500
k=pcm_to_fft(x,0,513)
k=k.real
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(3))

x=np.fromfile(open('Molly.wav'),np.int16)
k=x[38*88200:(38*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=numpy.random.rand(l)
x=x*1000
k=pcm_to_fft(x,0,513)
k=k.real
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(3))

x=np.fromfile(open('House_Of_Gold.wav'),np.int16)
k=x[78*88200:(78*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('fake_you_out.wav'),np.int16)
k=x[68*88200:(68*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('Highway_to_Hell.wav'),np.int16)
k=x[52*88200:(52*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=np.fromfile(open('journey.wav'),np.int16)
k=x[82*88200:(82*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))

x=numpy.random.rand(l)
x=x*1000
k=pcm_to_fft(x,0,513)
k=k.real
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(3))

x=np.fromfile(open('born_to_run.wav'),np.int16)
k=x[0*88200:(0*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))



x=numpy.random.rand(l)
x=x*1000
k=pcm_to_fft(x,0,513)
k=k.real
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(3))

x=numpy.random.rand(l)
x=x*1000
k=pcm_to_fft(x,0,513)
k=k.real
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(3))

x=numpy.random.rand(l)
x=x*1000
k=pcm_to_fft(x,0,513)
k=k.real
#k=pcm_to_fft(x,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(3))

x=np.fromfile(open('Dear_Body.wav'),np.int16)
k=x[43*88200:(43*88200)+(2*l)]
k=numpy.delete(k,range(0,len(k),2))
k=pcm_to_fft(k,0,513)
k=k.real
#k=pcm_to_fft(k,0,450)
#f=k.real
#s=k.imag
#k=list(chain.from_iterable(izip(f,s)))
ds.addSample(k,(0))   




'''











