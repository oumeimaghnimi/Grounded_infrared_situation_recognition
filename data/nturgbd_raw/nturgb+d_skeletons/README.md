
https://www.kaggle.com/datasets/aidagh/nturgbd

Example: S018C001P008R001A061.skeleton

  skeleton_sequence = {}
161
skeleton_sequence['numFrame']          Nombre des frames in the same file 161--->161 objects
1    #can me multiple bodies in one frame #      for m in range(frame_info['numBody']):
       
        frame_info['numBody']                        1

72057594037932486 0 1 1 0 0 0 -0.1665949 -0.1316579 2
            'bodyID': 72057594037932486
            0 1 1 0 0 0 -0.1665949 -0.1316579 2

          'clipedEdges': 0
          'handLeftConfidence':  1
           'handLeftState': 1
          'handRightConfidence': 0
          'handRightState': 0
           'isResticted': 0
          'leanX': -0.1665949
           'leanY': -0.1316579
           'trackingState': 2

25
        body_info['numJoint'] = int(f.readline())

-0.3785567 -0.1246493 3.539238 217.2075 221.3835 863.005 604.0732 -0.1717494 0.0553951 0.9777052 -0.1073591 2

     body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]

               'x': -0.3785567
               'y':- 0.1246493
               'z': 3.539238
              'depthX': 217.2075
              'depthY': 221.3835
             'colorX':   863.005
             'colorY':  604.0732
             'orientationW':  -0.1717494
             'orientationX':   0.0553951
              'orientationY':  0.9777052
             'orientationZ':  -0.1073591 
             'trackingState': 2


 body_info = {}
all in same dictionary: 

         body_info of each body in one dictionary . append them to one large body_info = {}

                    body_info['jointInfo'].append(joint_info)  

         frame_info['bodyInfo'].append(body_info)

                       the body info of all bodies in frame_info: our case  there are  161 frames in same annotation file.


def read_skeleton_filter(file): for one frame

Full dictionary foe each skeleton file :
 skeleton_sequence= {
                               
                                   'numFrame': 161   #161 frames
                                   'frameInfo':[
                                                                                                                                 #frame_1
                                                      {  'numBody':1                                                                             #how many body
                                                        'bodyInfo':                                                                                    #regroups all bodies
                                                        [                                                                                                   
                                                          {                                                                                                       #body_info_1
                                                             'bodyID': 72057594037932486,                                            #you find same body_Id of persons over frames in one file and the other parameters  can change
                                                               'clipedEdges': 0
                                                               'handLeftConfidence':  1
                                                                'handLeftState': 1
                                                               'handRightConfidence': 0
                                                               'handRightState': 0
                                                              'isResticted': 0
                                                              'leanX': -0.1665949
                                                              'leanY': -0.1316579
                                                              'trackingState': 2

                                                               'numJoint' : 25
                                                               'jointInfo' : [                                                            #one joint info
                                                                              {   'x': -0.3785567
                                                                                  'y':- 0.1246493
                                                                                  'z': 3.539238
                                                                                  'depthX': 217.2075
                                                                                 'depthY': 221.3835
                                                                                 'colorX':   863.005
                                                                                'colorY':  604.0732
                                                                                'orientationW':  -0.1717494
                                                                               'orientationX':   0.0553951
                                                                               'orientationY':  0.9777052
                                                                              'orientationZ':  -0.1073591 
                                                                              'trackingState': 2},   
                                                                             {}, {}, etc                                                #do for the other 24 joints

                                                                                   ]               
                                                                }, 

                                                           { } , {},  etc                 #body_info_2,   ....,  #body_info_4

                                                       ]

                                                    }

                                                   {}, {},                                     #frame_2,              #frame_numFrame 
                                                          ]
                         }



shape= (max_body,   seq_info['numFrame'],   num_joint, 3)
                data =  np.zeros(shape)
data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
--->data is an array

               m: N° of bodies        
                   n: nombre des frames 
                       j nombre des joints

   data[m, n, j, :] = [v['x'], v['y'], v['z']]
                   each sample correspond to one body(person)     #having same  bodyID': 72057594037932486,
                  s=   [n, j, [v['x'], v['y'], v['z']] 


(1, 161, 25, 3)

                                 frame_1=n                                                                             frame-2                         frame_3             frame_4                  frame_161
 
 m   bodyID_1    joint_1: [v_1['x'], v_1['y'], v_1['z']]  # j
                             joint_2: [v_2['x'], v_2['y'], v_2['z']]
                             joint_3: [v_3['x'], v_3['y'], v_3['z']]
                            ........
                             joint_25: [v_25['x'], v_25['y'], v_25['z']]

   bodyID_2
                           ........


   bodyID_3
                          ........




s.sum(-1)=  we will sum over the three dimensions --> we will obtain vector of dimension 25 ( joints)                # this is normalling done for all frame
S.sum(-1).sum(-1): sum over numeo of joints   #this is normalling done for all frame
--> we obtain vector of dimension Number of frames.: each frame has only sum joint
           index = s.sum(-1).sum(-1) != 0            output true or valse over al frames.  # select valid frames : 


def get_nonzero_std(s):  # tvc
       index = s.sum(-1).sum(-1) != 0
       s = s[index]

      if len(s) != 0:
          s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels                    #   we obtain sample  vector having standard deviation for every validated frame
     else:
          s = 0
    return  s


energy = np.array([get_nonzero_std(x) for x in data])              # for every bodyIDs  get standard deviarion vector of all frames
index = energy.argsort()[::-1][0:max_body_true]                       # search for sorting index -->  0, 1, 2 over frames for all bodyIDs
 data = data[index]                                                                             #data is sorted according to enery (standard deviation) over frames

 data = data.transpose(3, 1, 2, 0)                          #[ x,y, z], frames,    n°joint, body_id                                             

                                      

                                                                                                                 frame_1=n                                                                                                                              frame-2                         frame_3             frame_4                  frame_161
 
   
                                                                                               bodyID_1                                       bodyID_2

                                                                              joint 1   [v_1['x'], v_1['y'], v_1['z']] 

                                                                              joint_2: [v_2['x'], v_2['y'], v_2['z']]

                                                                             joint_3: [v_3['x'], v_3['y'], v_3['z']]
                                           
                             
                                                                           ........



                                                                             ........

https://github.com/shahroudy/NTURGB-D
Samples with missing skeletons:
302 samples in "NTU RGB+D" dataset and 535 samples in "NTU RGB+D 120" dataset have missing or incomplete skeleton data. If you are working on skeleton-based analysis, please ignore these files in your training and testing procedures.
https://github.com/shahroudy/NTURGB-D/blob/master/Matlab/NTU_RGBD_samples_with_missing_skeletons.txt

https://github.com/shahroudy/NTURGB-D/blob/master/Matlab/NTU_RGBD120_samples_with_missing_skeletons.txt

    for filename in os.listdir(data_path):
        if filename in ignored_samples:                              #ignore it by not processing it
            continue

Examples of skeleton file in https://www.kaggle.com/datasets/aidagh/nturgbd are    ~ .skeleton                 
                                                  S018C001P008R001A061.skeleton



Each file/folder name in both datasets is in the format of SsssCcccPpppRrrrAaaa (e.g., S001C002P003R002A013), 
in which sss is the setup number, ccc is the camera ID, ppp is the performer (subject) ID, rrr is the replication number (1 or 2), and aaa is the action class label.

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]

only select file corresponding to subject from list of  subjects(performers) and list of cameras Ids  for training : use it for splitting under training part or evaluation part.


        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

          if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

If chosen for training, append it the list of training samples with  its  annotation (here action classe)in different  list

    sample_name = []
    sample_label = []

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)


wb   :The wb mode will open the file for writing binary data.


    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)




example json dumping:

           with open(json_file, "w") as f:
                  json_str = json.dumps(res_file)
                  f.write(json_str)



https://docs.python.org/3/library/pickle.html           #see also GeotrackNet                              #we save filename and the class as pkl
example pickle dumping:

          with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
                pickle.dump((sample_name, list(sample_label)), f)


We have previously:


data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))


max_body_true = 2

 energy = np.array([get_nonzero_std(x) for x in data])
 index = energy.argsort()[::-1][0:max_body_true]
data = data[index]

data = data.transpose(3, 1, 2, 0)                          #[ x,y, z], frames,    n°joint, body_id         

Now  we will build an array fp:

 fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true=2), dtype=np.float32)



                                                                                                      frame_1=n                                                                                                                              frame-2                         frame_3             frame_4                  frame_161
 
   
                                                                                               bodyID_1                                       bodyID_2                 max_body=max_body_kinect

                                                                              joint 1   [v_1['x'], v_1['y'], v_1['z']] 
   filename_1   
                                                                              joint_2: [v_2['x'], v_2['y'], v_2['z']]

                                                                             joint_3: [v_3['x'], v_3['y'], v_3['z']]
                                                                             ........
                                                                             joint_25: [v_3['x'], v_3['y'], v_3['z']]
                                                                         

 filename_2   

                                                                             ........

 filename_len(sample_label)

 read_xyz function


max_body_kinect = 4
 max_body_true= 2


 fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true=2), dtype=np.float3
    for i, s in enumerate(tqdm(sample_name)):
        data = read_xyz(os.path.join(data_path, s), max_body=max_body_kinect, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data                          #0:data.shape[1]: get free from 0 to 161 icones for frames


from data_gen.preprocess import pre_normalization 

                                                 ## save the skelton annotation  array:  #good


fp = pre_normalization(fp)
np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)



out_path is just a string for renaming  pkl and array files


 with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:               #   benchmark = ['xsub', 'xview']       ,    for b in benchmark:                     out_path=  arg.out_folder/ b -->arg.out_folder/xsub,     arg.out_folder/ xview
        pickle.dump((sample_name, list(sample_label)), f)
                                                                                                                                                                                                                                                         arg.out_folder/xsub/train__label.pkl                                          arg.out_folder/ xview/train__label.pkl 
                                                                                                                                                                                                                                                         arg.out_folder/xsub/val_label.pkl                                               arg.out_folder/ xview/val__label.pkl 
                                                                                                                                                                   
np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)

                                                                                                                                                                                                                                                       arg.out_folder/xsub/train_data_joint.npy                                     arg.out_folder/ xview/rain_data_joint.npy
                                                                                                                                                                                                                                                      arg.out_folder/xsub/val_data_joint.npy                                          arg.out_folder/ xview/val_data_joint.npy 



training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]




In summary ntu_gendata.py will generate:



                         arg.out_folder/xsub/train__label.pkl                                          arg.out_folder/ xview/train__label.pkl 
                           arg.out_folder/xsub/val_label.pkl                                               arg.out_folder/ xview/val__label.pkl 
                                                                                                                                                                   


                        arg.out_folder/xsub/train_data_joint.npy                                     arg.out_folder/ xview/rain_data_joint.npy
                             arg.out_folder/xsub/val_data_joint.npy                                          arg.out_folder/ xview/val_data_joint.npy 











https://www.guru99.com/python-file-readline.html:

You have a file demo.txt, and when readline() is used, it returns the very first line from demo.txt.
demo.txt:
    Testing - FirstLine
    Testing - SecondLine
    Testing - Third Line
    Testing - Fourth Line
    Testing - Fifth Line

Step 1)
First, open the file using the file open() method, as shown below:
    myfile = open("demo.txt", "r")

The open() method takes the first parameter as the name of the file, and the second parameter is the mode is while you want to open. Right now, we have used “r”, which means the file will open in read mode.

             Mode 	Description
              R               This will open() the file in read mode.
             W               Using w, you can write to the file.
             a	       Using a with open() will open the file in write mode, and the contents will be appended at the end.
            rb               The rb mode will open the file for binary data reading.
            wb                The wb mode will open the file for writing binary data.


Use the readline() method to read the line from the file demo.txt as shown below:
              myline = myfile.readline()
             print(myline)
Once the reading is done, close the file using close() method as shown below:
            myfile.close()



Read a File Line-by-Line in Python:
The readline() method helps to read just one line at a time, and it returns the first line from the file given.
to read all the lines from the file given:
           Save the file demo.txt and use the location of demo.txt inside open() function.
          Using readline() inside while-loop will take care of reading all the lines present in the file demo.txt.


           myfile = open("demo.txt", "r")
           myline = myfile.readline()
           while myline:
                   print(myline)
                  myline = myfile.readline()
           myfile.close()   



or 


       myfile = open("test.txt", "r")
       for line in myfile:
            print(line)
           myfile.close()   


output:
          Line No 1
        Line No 2
       Line No 3
        Line No 4
        Line No 5


difference between  myfile.readline() and myfile.readlines()_How to read all lines in a file at once?

          myfile = open("test.txt", "r")
          mylist = myfile.readlines()
         print(mylist)
          myfile.close()
output: ['Line No 1\n', 'Line No 2\n', 'Line No 3\n', 'Line No 4\n', 'Line No 5']


https://www.w3schools.com/python/trypython.asp?filename=demo_ref_list_append : good for demos
fruits = ["apple", "banana", "cherry"]

fruits.append("orange")

print(fruits)







numpy.zeros(shape, dtype=float, order='C', *, like=None)
             s = (2,2)
            np.zeros(s)
output :
          array([[ 0.,  0.],
                       [ 0.,  0.]])

            np.sum([0.5, 1.5])
output:   
            2.0   


According to numpy doc, It is sum along the axis=-1 which in python is last axis. 
Here is a smaller example for easier representation:

arr = np.arange(2*3*4).reshape(2,3,4)
[  [  [ 0  1  2  3]
        [ 4  5  6  7]
        [ 8  9 10 11]  ]

 [     [12 13 14 15]
       [16 17 18 19]
       [20 21 22 23] ]   ]

print(arr.sum(0)) #same as print(arr.sum(-3))
[[12 14 16 18]
 [20 22 24 26]
 [28 30 32 34]]

print(arr.sum(1)) #same as print(arr.sum(-2))
[[12 15 18 21]
 [48 51 54 57]]

print(arr.sum(2)) #same as print(arr.sum(-1))
[[ 6 22 38]
 [54 70 86]]

     the last culum s is with 3 channels
       B.sum(-1): row-sum of B 
       B.sum(0):  column sum 

https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
numpy.argsort:
              numpy.argsort(a, axis= - 1, kind=None, order=None)
                     Returns the indices that would sort an array.
               Perform an indirect sort along the given axis using the algorithm specified by the kind keyword.
              It returns an array of indices of the same shape as a that index data along the given axis in sorted order.

    Example:                    
                         x = np.array([3, 1, 2])
                         np.argsort(x)
output:            array([1, 2, 0]) 
                   2    3  1

numpy.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<no value>, *, where=<no value>)[source]
Compute the standard deviation along the specified axis.


The strip() method returns a copy of the string by removing both the leading and the trailing characters (based on the string argument passed).

             message = '     Learn Python  '

            # remove leading and trailing whitespaces
             print('Message:', message.strip())

          # Output: Message: Learn Python


Where in the text is the word "welcome"?:
txt = "Hello, welcome to my world."

x = txt.find("welcome")

print(x) 
output : 7

There are fundamental differences between the pickle protocols and JSON (JavaScript Object Notation):

          JSON is a text serialization format (it outputs unicode text, although most of the time it is then encoded to utf-8), while pickle is a binary serialization format;

          JSON is human-readable, while pickle is not;

         JSON is interoperable and widely used outside of the Python ecosystem, while pickle is Python-specific;

         JSON, by default, can only represent a subset of the Python built-in types, and no custom classes; pickle can represent an extremely large number of Python types (many of them automatically, by clever usage of Python’s introspection facilities; complex cases can be tackled by 
               implementing specific object APIs);

        Unlike pickle, deserializing untrusted JSON does not in itself create an arbitrary code execution vulnerability.
       
        The json module: a standard library module allowing JSON serialization and deserialization.



pickle.dump(obj, file, protocol=None, *, fix_imports=True, buffer_callback=None)
        Write the pickled representation of the object obj to the open file object file. 
       This is equivalent to Pickler(file, protocol).dump(obj).

         Arguments file, protocol, fix_imports and buffer_callback have the same meaning as in the Pickler constructor.

        Changed in version 3.8: The buffer_callback argument was added.


pickle.dumps(obj, protocol=None, *, fix_imports=True, buffer_callback=None)
            Return the pickled representation of the object obj as a bytes object, instead of writing it to a file.
           Arguments protocol, fix_imports and buffer_callback have the same meaning as in the Pickler constructor.
           Changed in version 3.8: The buffer_callback argument was added.





