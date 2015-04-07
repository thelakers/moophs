#key functions
#04-01-2015
#import numpy as np

"""
input:
u_ids:     all the users
neighbors: object holds a neighbors matrix
output: pseudo-friends and pseudo-strangers
functions:
get_friends_strangers: generate pseudo-friends and pseudo-strangers  
"""

def get_friends_strangers(u_ids, neighbors):
    f_set = {}
    t_set = {}
    for u_id in u_ids:
        #get the neighbors of u_id
        nbs = neighbors.user_neighborhood(u_id)       
        if nbs:
            if len(nbs) < 500:
                continue

            my_friends = []
            my_strangers = []
            my_real_t = []
            my_real_f = []

            #assumption: Friends may have higher similarity.
            #            my_friends = nbs[2:402:2] is used to avoid
            #            choosing a friend who has extremely low similarity.
            #            Remove the users in top-2 similarity.
            #            !You can also use other strategies to generate friends and strangers 
            my_friends = nbs[2:402:2] 

            #assumption: Strangers may have wild similarity, but not too high.
            #            my_strangers = nbs[41::2] is used to preclude
            #            a stranger who has very high similarity
            my_strangers = nbs[41::2]
            
            #permutation
            seed = random.randint(1,10000000)
            rng1 = np.random.RandomState(seed)
            t_ids = rng1.permutation(len(my_strangers))
            
            for i in t_ids:
                my_real_t.append(my_strangers[i])

            seed = random.randint(1,10000000)
            rng2 = np.random.RandomState(seed)
            f_ids = rng2.permutation(len(my_friends))
            
            for j in f_ids:
                my_real_f.append(my_friends[j])

        f_set[u_id] = my_real_f
        t_set[u_id] = my_real_t

    return f_set,t_set

#-----------------------------------------------------------------------

"""
input:
u_id:          user id
i_id:          item id
friends_num:   how many friends will be used in prediction
strangers_num: how many strangers will be used in prediction
alpha:         prediction = alpha*friends_prediction + (1-alpha)*strangers_prediction
output:
prediction = alpha*friends_prediction + (1-alpha)*strangers_prediction
functions:
estimate_preference: the calulation is based on the friends and strangers obtained from get_friends_strangers
get_user_mean_preference: get mean preference of a user
get_friends_delta:  output->
    __                    _
    \    q_(f,b)*(r_(f,b)-r_(f))*w_(u,f)
    /_F
    ____________________________________
     __                
    \    q_(f,b)*w_(u,f)
    /_F
get_strangers_delta: output->
     __                   _
    \    q_(t,b)*(r_(t,b)-r_(t))
    /_T
    ____________________________
     __                
    \    q_(t,b)
    /_T
"""

def estimate_preference(self, u_id, i_id, friends_num, strangers_num, alpha):
    pref = 0.0
    #get mean preference of a user
    mean_ru = get_user_mean_preference(u_id) 
    if mean_ru < 0:
        #impossible!
        raise "fatal error!"

    #get friends delta of a user
    f_delta = get_friends_delta(u_id,i_id,friends_num)
           
    #calulate friend prediction
    f_pref = mean_ru + f_delta
    
    if alpha < 1:
        t_delta = get_strangers_delta(u_id,i_id,strangers_num)
        t_pref = mean_ru + t_delta
        pref = alpha*f_pref + (1.0-alpha)*t_pref
    else:
        pref = f_pref

    return pref


