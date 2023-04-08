import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


from utils import label_map_util
from utils import visualization_utils as vis_util

####################

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    except:
        print('DALAS************')
'''
####################


MODEL_NAME='inference_graph'
PATH_TO_FROZEN_GRAPH = MODEL_NAME+'/frozen_inference_graph.pb'
PATH_TO_LABELS='training/labelmap.pbtxt'

detection_graph=tf.Graph()
with detection_graph.as_default():
    od_graph_def=tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph=fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)

coord_x1=0
coord_y1=0
coord_x2=0
coord_y2=0
cambiante=0
def get_puntos(event,x,y,flags,param):
    global coord_x1
    global coord_y1
    global coord_x2
    global coord_y2
    global cambiante
    if event==cv2.EVENT_LBUTTONDOWN:
        if cambiante==0:
            coord_x1=x
            coord_y1=y
            cambiante=1
        else:
            coord_x2=x 
            coord_y2=y 
            cambiante=2 


def run_inference_for_single_image(image, graph):
    if 'detection_masks' in tensor_dict:
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'],[0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'],[0])
        real_num_detection =tf.cast(tensor_dict['num_detections'][0],tf.int32)
        detection_boxes = tf.slice(detection_boxes,[0,0],[real_num_detection,-1])
        detection_masks = tf.slice(detection_masks,[0,0,0],[real_num_detection,-1,-1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks,detection_boxes,image.shape[0],image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed,0.5),tf.uint8)
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed,0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    
    output_dict = sess.run(tensor_dict,feed_dict={image_tensor:np.expand_dims(image,0)})
    output_dict['num_detections']=int(output_dict['num_detections'][0])
    output_dict['detection_classes']=output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes']=output_dict['detection_boxes'][0]
    output_dict['detection_scores']=output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks']=output_dict['detection_masks'][0]
    return output_dict

import cv2
#cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture('2.ogv')

with detection_graph.as_default():
    with tf.Session() as sess:
        ops=tf.get_default_graph().get_operations()
        all_tensor_names={output.name for op in ops for output in op.outputs}
        tensor_dict ={}
        for key in [
            'num_detections','detection_boxes','detection_scores',
            'detection_classes','detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key]=tf.get_default_graph().get_tensor_by_name(tensor_name)
        _,imagen=cap.read()
        imagen=cv2.resize(imagen,(1000,700))
        cv2.namedWindow("Imagen")
        cv2.setMouseCallback("Imagen",get_puntos)
        while True:
            cv2.imshow("Imagen",imagen)
            k=cv2.waitKey(1) & 0xFF
            if cambiante==2:
                cambiante=0
                break
            if k==27:
                break
        cv2.destroyAllWindows()
        print("**********************")
        print("X1: ",coord_x1,"\tY1: ",coord_y1)
        print("X2: ",coord_x2,"\tY2: ",coord_y2)
        contador=0
        matriz_existencia=np.ones(8)
        matriz_existencia_2=np.ones(8)
        matriz_existencia_final=np.ones(8)
        cantidad_cuadros=0
        while True:
            ret, image_np=cap.read()
            image_np=cv2.resize(image_np,(1000,700))
            image_np_expanded = np.expand_dims(image_np,axis=0)
            output_dict=run_inference_for_single_image(image_np,detection_graph)
            '''
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                min_score_thresh=.99,                
                line_thickness=8)
            '''
            (x,)=output_dict['detection_classes'].shape
            m,n,_=image_np.shape

            gray=0.3*image_np[:,:,2] + 0.59*image_np[:,:,1]+0.11*image_np[:,:,1]
            gray_matrix=np.zeros((m,n,3), dtype=np.uint8)
            gray_matrix[:,:,0]=gray
            gray_matrix[:,:,1]=gray
            gray_matrix[:,:,2]=gray

            image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            laplacian=cv2.Canny(image_np,30,150)

            igualdad_valores=np.equal(matriz_existencia_2,matriz_existencia) # IGUALAR VALORES DE DETECCION
            matriz_existencia_2=matriz_existencia
            if(igualdad_valores.all()):
                cantidad_cuadros=cantidad_cuadros+1
            else:
                cantidad_cuadros=0
            if cantidad_cuadros>10: #MAYOR A N REPETICIONES
                matriz_existencia_final=matriz_existencia
                print("CANTIDAD CUADROS: ",cantidad_cuadros)
  
            for indice0 in range(8):
                pos_existencia=indice0+1
                valor_existencia=matriz_existencia_final[indice0]
                cuadrito_d=(coord_x2-coord_x1)*0.1
                if pos_existencia<5:
                    xx=coord_x1-int(((4-pos_existencia)*2+1.5)*cuadrito_d)
                    if valor_existencia==1:
                        print(f'DIENTE {pos_existencia} ✓')
                        cv2.rectangle(image_np,(xx,m-10),(xx-int(cuadrito_d),m-10-int(cuadrito_d)),(0,255,0),-1)
                    else:
                        print(f"DIENTE {pos_existencia} x")
                        cv2.rectangle(image_np,(xx,m-10),(xx-int(cuadrito_d),m-10-int(cuadrito_d)),(0,0,255),-1)
                else:
                    xx=coord_x1+int(((pos_existencia-5)*2+0.5)*cuadrito_d)
                    if valor_existencia==1:
                        print(f'DIENTE {pos_existencia} ✓')
                        cv2.rectangle(image_np,(xx,m-10),(xx-int(cuadrito_d),m-10-int(cuadrito_d)),(0,255,0),-1)
                    else:
                        print(f"DIENTE {pos_existencia} x")
                        cv2.rectangle(image_np,(xx,m-10),(xx-int(cuadrito_d),m-10-int(cuadrito_d)),(0,0,255),-1)

            campana=1000
            a1=0
            b1=0
            c1=0
            d1=0
            num_dientes=0
            pos_dientes=np.zeros((1,2))
            longi_dientes=np.zeros(1)

            for k in range(x): 
                if str(output_dict['detection_classes'][k]) =='2' and float(output_dict['detection_scores'][k])>=0.80:
                    a_str,b_str,c_str,d_str=output_dict['detection_boxes'][k]
                    a1=int(a_str*m)
                    c1=int(c_str*m)
                    b1=int(b_str*n)
                    d1=int(d_str*n)
                    if c1<=int(coord_y2): #toma menos de la mitad de la pantalla
                        campana =k

            if campana != 1000:
                for i in range(x):
                    score_i=float(output_dict['detection_scores'][i])
                    class_i=str(output_dict['detection_classes'][i])
                    a,b,c,d = output_dict['detection_boxes'][i]
                    a=int(a*m)
                    c=int(c*m)
                    b=int(b*n)
                    d=int(d*n)
                    area=(d-b)*(c-a)
                    if score_i >= 0.40 and class_i =='1' and a>=a1 and c<=c1 and b<=int(coord_x2) and area<=35000:
                        cv2.rectangle(image_np,(b,a),(d,c),(255,0,0),3) #recuadro cada diente
                        #cv2.putText(image_np,str(area),(b,a),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0,0), 2, cv2.LINE_AA) #area diente
                        #pos_dientes=np.append(pos_dientes,[[int((b+d)*.5),int((a+c)*.5)]],axis=0) #AL MEDIO
                        pos_dientes=np.append(pos_dientes,[[int((b+d)*.5),c]],axis=0)
                cv2.rectangle(image_np,(b1,a1),(d1,c1),(0,255,0),3) #cuadro grande
                num_dientes=pos_dientes.shape[0]-1
                num_dientes_odenado=pos_dientes[pos_dientes[:,0].argsort()] #ordenando de izquiera a derecha
                for ii in range(num_dientes):
                    numero_diente=ii+1
                    punta_x1=int(num_dientes_odenado[numero_diente][0])
                    punta_y1=int(num_dientes_odenado[numero_diente][1])
                    #cv2.putText(image_np,str(numero_diente),(punta_x1,punta_y1),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2, cv2.LINE_AA) #ESCRIBIR EL NUMERO DE CADA DIENTE
                    if numero_diente<num_dientes and num_dientes>1:
                        punta_x2=int(num_dientes_odenado[numero_diente+1][0])
                        punta_y2=int(num_dientes_odenado[numero_diente+1][1])
                        #cv2.line(image_np,(punta_x1,punta_y1),(punta_x2,punta_y2),(0,0,255),4) #linea entre puntos
                        dist_diente=punta_x2-punta_x1
                        #print("Distancia ",numero_diente,": ",dist_diente)
                        longi_dientes=np.append(longi_dientes,dist_diente)
                longi_dientes_ordenado=np.sort(longi_dientes)
                try:
                    promedio_diente=longi_dientes_ordenado[1]
                except:
                    promedio_diente=0
                matriz_existencia=np.zeros(8)
                for pp in range(num_dientes):
                    numero_diente=pp+1
                    punta_x1=int(num_dientes_odenado[numero_diente][0])
                    punta_y1=int(num_dientes_odenado[numero_diente][1])
                    dd=coord_x1-punta_x1
                    dd_abs=abs(dd)
                    preciso=0.4
                    if dd>0:
                        if abs(dd_abs-3.5*promedio_diente)<preciso*promedio_diente:
                            cv2.putText(image_np,str(1),(punta_x1,punta_y1),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2, cv2.LINE_AA)
                            matriz_existencia[0]=1                    

                        if abs(dd_abs-2.5*promedio_diente)<preciso*promedio_diente:
                            cv2.putText(image_np,str(2),(punta_x1,punta_y1),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2, cv2.LINE_AA)
                            matriz_existencia[1]=1 

                        if abs(dd_abs-1.5*promedio_diente)<preciso*promedio_diente:
                            cv2.putText(image_np,str(3),(punta_x1,punta_y1),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2, cv2.LINE_AA)
                            matriz_existencia[2]=1 

                        if abs(dd_abs-0.5*promedio_diente)<preciso*promedio_diente:
                            cv2.putText(image_np,str(4),(punta_x1,punta_y1),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2, cv2.LINE_AA)
                            matriz_existencia[3]=1 

                    else:
                        if abs(dd_abs-3.5*promedio_diente)<preciso*promedio_diente:
                            cv2.putText(image_np,str(8),(punta_x1,punta_y1),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2, cv2.LINE_AA)
                            matriz_existencia[7]=1 

                        if abs(dd_abs-2.5*promedio_diente)<preciso*promedio_diente:
                            cv2.putText(image_np,str(7),(punta_x1,punta_y1),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2, cv2.LINE_AA)
                            matriz_existencia[6]=1 

                        if abs(dd_abs-1.5*promedio_diente)<preciso*promedio_diente:
                            cv2.putText(image_np,str(6),(punta_x1,punta_y1),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2, cv2.LINE_AA)
                            matriz_existencia[5]=1 

                        if abs(dd_abs-0.5*promedio_diente)<preciso*promedio_diente:
                            cv2.putText(image_np,str(5),(punta_x1,punta_y1),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2, cv2.LINE_AA)
                            matriz_existencia[4]=1 
                '''            
                for indice in range(8):
                    pos_existencia=indice+1
                    valor_existencia=matriz_existencia[indice]
                    cuadrito_d=(coord_x2-coord_x1)*0.1
                    if pos_existencia<5:
                        xx=coord_x1-int(((4-pos_existencia)*2+1.5)*cuadrito_d)
                        if valor_existencia==1:
                            print(f'DIENTE {pos_existencia} ✓')
                            cv2.rectangle(image_np,(xx,m-10),(xx-int(cuadrito_d),m-10-int(cuadrito_d)),(0,255,0),-1)
                        else:
                            print(f"DIENTE {pos_existencia} x")
                            cv2.rectangle(image_np,(xx,m-10),(xx-int(cuadrito_d),m-10-int(cuadrito_d)),(0,0,255),-1)
                    else:
                        xx=coord_x1+int(((pos_existencia-5)*2+0.5)*cuadrito_d)
                        if valor_existencia==1:
                            print(f'DIENTE {pos_existencia} ✓')
                            cv2.rectangle(image_np,(xx,m-10),(xx-int(cuadrito_d),m-10-int(cuadrito_d)),(0,255,0),-1)
                        else:
                            print(f"DIENTE {pos_existencia} x")
                            cv2.rectangle(image_np,(xx,m-10),(xx-int(cuadrito_d),m-10-int(cuadrito_d)),(0,0,255),-1)
                '''
                #print(contador)
                contador=contador+1
                print("*******************************")
            

            dientes_str="Dientes: "+str(num_dientes)
            #cv2.putText(image_np,dientes_str,(50,50),cv2.FONT_HERSHEY_SIMPLEX,3, (0, 0,255), 2, cv2.LINE_AA) #TEXTO NUMERO DE DIENTES
            #cv2.line(image_np,(int((coord_x1+coord_x2)*0.5),coord_y1),(int((coord_x1+coord_x2)*0.5),coord_y2),(255,0,0),4) #LINEA AL CENTRO
            
            '''
            for i in range(x):
                score_i=output_dict['detection_scores'][i]
                class_i=output_dict['detection_classes'][i]
                f_i=float(score_i)
                if f_i >= 0.99:
                    #print ("Score: "+ str(f_i)) 
                    #print ("Clase: "+str(output_dict['detection_classes'][i]))
                    a,b,c,d = output_dict['detection_boxes'][i]
                    a=int(a*m)
                    c=int(c*m)
                    b=int(b*n)
                    d=int(d*n)
                    #print("a: "+str(a))
                    #print("b: "+str(b))
                    #print("c: "+str(c))
                    #print("d: "+str(d))
                    cv2.rectangle(image_np,(b,a),(d,c),(255,0,0),3)
                    cv2.putText(image_np,str(class_i),(b,a-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

                    #image_np[a:c,b:d,:] = gray_matrix[a:c,b:d,:] ###----------->>>>>

                    #image_np[a:c,b:d,0] = laplacian[a:c,b:d]
                    #image_np[a:c,b:d,1] = laplacian[a:c,b:d]
                    #image_np[a:c,b:d,2] = laplacian[a:c,b:d]
                    
            '''
            #cv2.putText(image_np, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('GETS DETECTION', cv2.resize(image_np,(800,600))) 
            #cv2.imshow('object_detection', image_np)          
            #cv2.imshow('object_detection', gray_matrix)     
            if cv2.waitKey(25) & 0xFF ==ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break