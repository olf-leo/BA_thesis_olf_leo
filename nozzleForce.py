import pybullet as p
import time
import math
import pybullet_data
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from stl import mesh
from scipy.spatial.transform import Rotation as R

def visualize_results(mesh_path, position, orientation, ray_results, wsf):
    # 1. Load and Scale STL
    stl_mesh = mesh.Mesh.from_file(mesh_path)
    # stl_mesh.vectors is an array of [triangles, vertices, xyz]
    vectors = stl_mesh.vectors * wsf 

    # 2. Apply PyBullet Rotation and Translation
    # PyBullet orientation is [x, y, z, w]
    rot = R.from_quat(orientation)
    
    # Flatten to transform all vertices, then reshape back
    points = vectors.reshape(-1, 3)
    rotated_points = rot.apply(points)
    translated_points = rotated_points + np.array(position)
    final_vectors = translated_points.reshape(-1, 3, 3)

    # 3. Extract Hit Points
    hit_points = []
    for i in range(1, len(ray_results)):
        if ray_results[i][0] > -1:
            hit_points.append(ray_results[i][3])
    hit_points = np.array(hit_points)

    #print(hit_points)

    # 4. Create Figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 5. Plot Mesh with Edges
    # edgecolor='black' displays the wireframe/edges
    poly = mplot3d.art3d.Poly3DCollection(final_vectors, alpha=0.15)
    poly.set_facecolor('royalblue')
    poly.set_edgecolor('black') 
    poly.set_linewidth(0.3)
    ax.add_collection3d(poly)

    # 6. Plot Hit Points
    if len(hit_points) > 0:
        ax.scatter(hit_points[:, 0], hit_points[:, 1], hit_points[:, 2], 
                   color='red', s=10, label='Ray Hits', depthshade=False)

    # 7. Center 0,0,0 and offset Z to the bottom plane
    # Find global bounds to set axes limits
    all_dims = final_vectors.reshape(-1, 3)
    max_range = np.array([all_dims[:,0].max()-all_dims[:,0].min(), 
                          all_dims[:,1].max()-all_dims[:,1].min(), 
                          all_dims[:,2].max()-all_dims[:,2].min()]).max() / 2.0

    mid_x = (all_dims[:,0].max() + all_dims[:,0].min()) * 0.5
    mid_y = (all_dims[:,1].max() + all_dims[:,1].min()) * 0.5
    min_z = all_dims[:,2].min()

    # Set limits so object is centered in XY and sits on the Z=0 floor
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(min_z, min_z + max_range * 2) 

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Analysis: {len(hit_points)} Hits Detected')
    
    plt.legend()
    plt.show()

def calc_force(
        name_obj, 
        start_pos, 
        cw, 
        nozzle_distance, 
        nozzle_diameter, 
        nozzle_pressure,
        ray_number,
        flow_mode,
        print_results,
        graph,
        use_gui):
   
    start_time = time.time()


    #name_obj='Qf4i'
    #name_obj='1dx1h_disc'


    #use_gui = True
    wsf = 0.001  #short for world scaling factor, so 1 unit is 1 mm

    if (use_gui):
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setGravity(0, 0, -10)

    # Set camera Settings
    if (use_gui):
        p.resetDebugVisualizerCamera(
            cameraDistance=150.0*wsf,         # Increase this value to zoom further out
            cameraYaw=-35.0,             # Optional: Horizontal angle
            cameraPitch=-50,          # Optional: Vertical angle
            cameraTargetPosition=[0, 0, 0] # Optional: The point in the world to look at
        )

    # Import Plane
    planeId = p.loadURDF("plane.urdf", basePosition=[0,0,0])

    # Import STL
    #startPos = [-2.0*wsf,-10.0*wsf,15.0*wsf]
    #start_pos = [0.0*wsf,0.0*wsf,15.0*wsf]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    mesh_file_path = '/Users/leonardolfens/Desktop/Python_Match/pybullet/STLs/'+name_obj+'.stl' 

    # sloppy compensation for weird model
    workpiece_scale = [1.0*wsf,1.0*wsf,1.0*wsf]
    #com_offset = [40.0*wsf, 40.0*wsf, 10*wsf]
    com_offset = [0.0*wsf, 0.0*wsf, 0.0*wsf]

    # 1. Create the collision shape
    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=mesh_file_path,
        meshScale=workpiece_scale,
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH
    )

    # 2. Create the visual shape 
    visual_shape_id = -1
    if (use_gui):
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=mesh_file_path,
            meshScale=workpiece_scale,
            rgbaColor=[0.5, 0.5, 0.5, 1] # Optional: Set color (R, G, B, Alpha)
        )

    # 3. Combine them into a physics body
    # Mass > 0 creates a dynamic object; mass = 0 creates a static object (like the plane)
    object_mass = 1.0 
    object_id = p.createMultiBody(
        baseMass=object_mass,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        baseInertialFramePosition=com_offset, 
        basePosition=start_pos,
        baseOrientation=startOrientation
    )

    # reduce bouncing
    p.changeDynamics(
        bodyUniqueId=object_id, 
        linkIndex=-1,  
        restitution=0.1,
        linearDamping=0.9,   
        angularDamping=0.9
    )

    p.changeDynamics(
        bodyUniqueId=planeId, 
        linkIndex=-1,  # -1 refers to the base link of the object/plane
        restitution=0.1
    )

    #print(f"Loaded object with ID: {object_id}")

    # Define Ray origin and end points
    ray_number_proportional_to_area = True

    if not (ray_number_proportional_to_area):
        circle_resolution = 30
        circle_number = 12
        ray_spread = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]           #[0.5,1,1.5,2]
        ray_number = circle_resolution*circle_number+1

    ray_height = 100*wsf
    #nozzle_distance = 25*wsf
    nozzle_spread = 20 #[Degrees]
    d1 = nozzle_diameter*wsf #nozzle diameter [mm]
    ray_height += d1/math.tan(nozzle_spread*math.pi/180/2)
    ray_angle = 0.0
    circle_index = 0

    #ray_number = 300
    cone_diameter = ray_height*math.tan(nozzle_spread*math.pi/180/2)*2
    cone_stepsize = cone_diameter/10
    circle_number = round(cone_diameter/cone_stepsize)
    cone_area = (cone_diameter/2)**2*math.pi
    circle_areas = []
    circle_areas.append((cone_stepsize/2)**2*math.pi)
    donut_areas = []
    donut_areas.append((cone_stepsize/2)**2*math.pi)
    ray_spread = []
    #ray_spread.append(cone_stepsize/4)
    circle_resolutions = []

    for i in range(1,circle_number):
        circle_areas.append(((cone_stepsize+i*cone_stepsize)/2)**2*math.pi)
        donut_areas.append(circle_areas[i]-circle_areas[i-1])
        #ray_spread.append(cone_stepsize/2*i-(cone_stepsize/2))
        

    for i in range(0,circle_number):
        circle_resolutions.append(round(ray_number*donut_areas[i]/cone_area))
        ray_spread.append(cone_stepsize/2*(i+1)-cone_stepsize/4)
    
    ray_number=sum(circle_resolutions)+1

    #print(ray_spread)
    #print(cone_diameter)
    #print(circle_areas)
    #print(donut_areas)
    #print(ray_spread)
    #print(circle_resolutions)
    #print(ray_number)

    rayFrom = []
    rayFrom.append([0.0*wsf, 0.0*wsf, -nozzle_distance*wsf-(d1/math.tan(nozzle_spread*math.pi/180/2))])
    rayTo = []
    rayTo.append([
    rayFrom[0][0],
    rayFrom[0][1],
    rayFrom[0][2]+ray_height
        ])

    #print(rayFrom[0])
    #print(nozzle_distance)
    #print(d1)
    #print(nozzle_spread)

    rayIds = []

    rayHitColor = [1, 0, 0]
    rayMissColor = [0, 1, 0]

    replaceLines = True

    rayIds.append(p.addUserDebugLine(rayFrom[0], rayTo[0], rayMissColor))

    for i in range(1, ray_number+1):
        rayFrom.append(rayFrom[0])
        rayTo.append([
            math.sin(ray_angle) * ray_spread[circle_index],
            math.cos(ray_angle) * ray_spread[circle_index],
            ray_height
        ])
        #print("index: "+str(circle_index)+", angle: "+str(ray_angle)+", sin: "+str(math.sin(ray_angle))+", cos: "+str(math.cos(ray_angle))+", spread. "+str(ray_spread[circle_index]))
        
        
        if (replaceLines):
            rayIds.append(p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor))
        else:
            rayIds.append(-1)


        if ray_angle >= (2*math.pi)/circle_resolutions[circle_index]*(circle_resolutions[circle_index]-1):
            ray_angle = 0
            circle_index += 1
        else:
            ray_angle = ray_angle + (2*math.pi)/circle_resolutions[circle_index]

    #print(donut_areas)
    #print(circle_resolutions)
    
    location_matrix = []
    rotation_matrix = []


    for i in range (500):
        p.stepSimulation()
        location_orientation = p.getBasePositionAndOrientation(object_id)
        location_matrix.append([
            round(location_orientation[0][0], 4), 
            round(location_orientation[0][1], 4),
            round(location_orientation[0][2], 4)
            ])
        rotation_matrix.append([
            round(location_orientation[1][0], 4), 
            round(location_orientation[1][1], 4),
            round(location_orientation[1][2], 4),
            round(location_orientation[1][3], 4),
            ])

        if (i>5 and location_matrix[i]==location_matrix[i-5] and rotation_matrix[i]==rotation_matrix[i-5]):
            #print('break!')
            object_location=location_matrix[i]
            object_rotation=rotation_matrix[i]
            break

        if (i>=499):
            object_location=location_matrix[i]
            object_rotation=rotation_matrix[i]

        if(use_gui):
            time.sleep(1./500.)



    if (not use_gui):
        timingLog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "rayCastBench.json")

    numSteps = 1
    if (use_gui):
        numSteps = 1


    for i in range(numSteps):
    #p.stepSimulation()
    #for j in range(8):
        #results = p.rayTestBatch(rayFrom, rayTo, j + 1)

        results = p.rayTestBatch(rayFrom, rayTo)

        #for i in range (10):
        #	p.removeAllUserDebugItems()

        if (use_gui and i<1):
            if (not replaceLines):
                p.removeAllUserDebugItems()

            for i in range(ray_number+1):
                hitObjectUid = results[i][0]

                if (hitObjectUid < 0):
                    hitPosition = [0, 0, 0]
                    p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor, replaceItemUniqueId=rayIds[i])
                else:
                    hitPosition = results[i][3]
                    p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor, replaceItemUniqueId=rayIds[i])
                    #print(rayIds[i], results[i][3])
                
        p.stepSimulation()
        #time.sleep(1.)

    #Calculate centricity coefficient
    Cc_method = 3    #1: ray based, 2: circle based, custom distribution, 3: cirlce based, gaussian distribution
    
    falloff_multiplier = []
    average = 0.5/circle_number

    mu = circle_number
    sigma = circle_number/3.4      #2.15 or 3.15

    for i in range (1, circle_number+1):
    
        if (i>1):
            falloff_multiplier.append(round((norm.cdf(i, loc=mu, scale=sigma)-norm.cdf(i-1, loc=mu, scale=sigma))/average, 5))
        else:
            falloff_multiplier.append(round(norm.cdf(i, loc=mu, scale=sigma)/average,5))

    Cc = 1.0 #Centricity coefficioent

    if (Cc_method == 2):
        falloff_multiplier = [0.25, 0.5, 0.75, 0.75, 1, 1.25, 1.5, 2, 2, 2]
    #for i in range 

    #falloff_multiplier = np.flip(falloff_multiplier)
    #print(falloff_multiplier)
    #print(sum(falloff_multiplier))
    hits_in_circle = []

    multiplier_total = 0.0
    average2 = 0.0
    current_ray = 0
    hits = 0

    for i in range (circle_number):
        hits = 0
        average2 += falloff_multiplier[circle_number-1-i]*circle_resolutions[i]
        for j in range (circle_resolutions[i]):
            current_ray+=1
            if (results[current_ray][0] > 0):
                hits += 1
                multiplier_total+= falloff_multiplier[circle_number-1-i]
                #print(falloff_multiplier[circle_number-1-i])
                #print(current_ray)
        hits_in_circle.append(hits)

    average2 = average2/ray_number
    #print(average2) 

    Cc = round(multiplier_total/sum(hits_in_circle)/average2, 5)
    

    #print(sum(hits_in_circle))

    hit_number = sum(hits_in_circle)
    multiplier_total = 0.0
    multiplier_balance = []
    #donut_area_percentage = []

    for i in range(circle_number):
        #donut_area_percentage.append(hits_in_circle[i]/hit_number)
        multiplier_balance.append(hits_in_circle[i]/circle_resolutions[i])
        multiplier_total += multiplier_balance[i]*falloff_multiplier[circle_number-1-i]

    if (Cc_method == 2 or Cc_method == 3):
        Cc = round(multiplier_total/sum(multiplier_balance), 5)

    #print(circle_resolutions)
    #print(falloff_multiplier)
    #print(donut_area_percentage)
    #print(sum(donut_area_percentage))

    hit_number = 0
    distance_sum = 0

    for i in range (1, ray_number+1):
        if (results[i][0] > 0):
            hit_number += 1
            distance_sum += results[i][3][2] - rayFrom[i][2] - d1/math.tan(nozzle_spread*math.pi/180/2)
        #print(round(results[i][3][2], 2))
        #print(rayFrom[i][2])
        #print(results[i][3][2] - rayFrom[i][2])
    #print(distance_sum)

    hit_fraction = hit_number/ray_number

    average_distance = 0
    if not (distance_sum == 0):
        average_distance = distance_sum/hit_number/wsf
    

    y = average_distance/1000 #distance from nozzle to object [m]
    r1 =y*math.tan(nozzle_spread*math.pi/180/2)+d1/2 #radius of projected circle
    Ast = r1**2*math.pi*hit_fraction #Anstroemflaeche [m^2]
    roh = 1.225 #density of air [Kg/m^3]
    p1 = 101325 #ambient air pressure [Pa]
    p0 = p1 + nozzle_pressure #nozzle pressure [Pa]
    gamma = nozzle_spread 
    #cw = 1.17 #drag coefficient [-]
    kappa = 1.4 #heat capacity ratio [-]
    #already established further up: d1 = 0.0035 #nozzle diameter [m]
    #Fw  = cw*Ast*p1*(kappa/(kappa-1))*(1-(p1/p0)**((kappa-1)/kappa))*(d1/(d1+2*y*math.sin(gamma*math.pi/180/2)))**2 #resistance force [N] 
    Fw  = cw*Ast*p1*(kappa/(kappa-1))*(p0-p1+1500)/550000*(d1/(d1+2*y*math.sin(gamma*math.pi/180/2)))**2 #trial and error
    #Fw  = cw*Ast*p1*(1-(p1/p0))*(d1/(d1+2*y*math.sin(gamma*math.pi/180/2)))**2#trial and error
    Fwb = cw*Ast*p1*(kappa/(kappa-1))*(1-(p1/p0)**(kappa/(kappa-1)))*(d1/(d1+2*y*math.sin(gamma*math.pi/180/2)))**2 #Bansman method used in Matlab, still contains mistake

    if(flow_mode):
        Fw = cw*roh/2*((nozzle_pressure*1000/(2*math.pi))**2)*Ast*(d1/(d1+2*y*math.sin(gamma*math.pi/180/2)))**2

    if (print_results):
        print('Object name: '+name_obj)
        print('Object location: '+str(object_location))
        print('Object rotation: '+str(object_rotation))
        print('average distance: '+str(round(average_distance, 2))+' mm')
        print('Number of Hits: '+str(hit_number)+' out of '+str(ray_number)+' ('+str(round(hit_fraction*100))+'%)')
        print('Centricity coefficient: '+str(Cc))
        print('Anstroemflaeche: '+str(Ast*1000000)+' mm^2')
        print('Resistance force: '+str(Fw)+' N (with Nozzle pressure '+str(p0)+' Pa)')

        workpiece_data = open('/Users/leonardolfens/Desktop/Python_Match/pybullet/Output_txt/'+name_obj+'_simulated_data.txt','w')
        workpiece_data.writelines("-------------------------------------------")
        workpiece_data.writelines('\n'+'Simulated_Data_'+name_obj+'\n')
        workpiece_data.writelines("-------------------------------------------"+'\n')
        workpiece_data.writelines('Object location: '+str(object_location)+'\n')
        workpiece_data.writelines('Object rotation: '+str(object_rotation)+'\n')
        workpiece_data.writelines('average distance: '+str(round(average_distance, 2))+' mm\n')
        workpiece_data.writelines('Number of Hits: '+str(hit_number)+' out of '+str(ray_number)+' ('+str(round(hit_fraction*100))+'%)\n')
        workpiece_data.writelines('Resistance force: '+str(Fw)+' N (with Nozzle pressure '+str(p0)+' Pa)\n')
        workpiece_data.close()

    

    if (not use_gui):
        p.stopStateLogging(timingLog)
        p.disconnect()
    else:
        while (p.isConnected()):
            p.stepSimulation()
            time.sleep(1./240.)

    if (graph):
        visualize_results(mesh_file_path, object_location, object_rotation, results, wsf)

    end_time = time.time()
    runtime = end_time-start_time

    if (print_results):
        print('Runtime: '+str(runtime*1000)+' ms')

    
    return Fw, Ast, hit_number, Cc
    


#calc_force('0_5dx1h_disc', [0,0,0], 1.17, 26, 4, 60000, 300, False, True, True, False)
#calc_force('4dx1h_disc', [0,0,0], 1.17, 26, 4, 2.88, 300, True, True, False, False)

