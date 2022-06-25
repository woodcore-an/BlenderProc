import blenderproc as bproc
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', help="Path to the bop datasets parent directory")
parser.add_argument('cc_textures_path', default="resources/cctextures", help="Path to downloaded cc textures")
parser.add_argument('output_dir', help="Path to where the final files will be saved")
parser.add_argument('--num_scenes', type=int, default=2000, help="How many scenes with 25 images each to generate")
args = parser.parse_args()

bproc.init()

# 加载bop对象到场景中
target_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'bottle'), model_type =  'proc', mm2m = True)

# 加载bop干扰对象
tless_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'tless'), model_type = 'cad', mm2m = True)
# lm_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'lm'), mm2m=True)

# 加载bop数据集的相机内参（相机固有参数）
bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(args.bop_parent_path, 'bottle'))

# 设置阴影和隐藏对象
# for obj in (target_bop_objs + tless_dist_bop_objs + lm_dist_bop_objs):
for obj in (target_bop_objs + tless_dist_bop_objs):
    obj.set_shading_mode('auto')
    obj.hide(True)

# 创建房间，从5个平面构建最小的2m x 2m x 2m的房间
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),  # bproc.object.create_primitive()创建一个新的原始网格对象，返回类型MeshObject
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
for plane in room_planes:
    # enable_rigidbody(active, collision_shape='CONVEX_HULL', collision_margin=0.001, collision_mesh_source='FINAL', mass=None, mass_factor=1, friction=0.5, angular_damping=0.1, linear_damping=0.04)
    # active: 如果为True则对象主动参与模拟，其关键帧将被忽略。如果为False，对象遵循其关键帧，仅充当障碍物，不受模拟的影响。！！！而这里是墙，所以不参与模拟，只充当障碍物，不受模拟影响
    # collision_shape: 模拟中对象的碰撞状。！！！这里为盒子
    # mass: 物体应具有的质量（以千克为单位），如果为None，则根据其边界框体积和给定的mass_factor计算质量
    # friction: 摩擦力，物体的运动阻力。
    # linear_damping: 随时间推移损失的线速度量
    # angular_damping: 随时间推移损失的角度量
    plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)  # enable_rigidbody()启用对象的刚体元素，使其参与物理模拟

# 从天花板上采样光的颜色和强度
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')  # 创建一个新的空材料light_material

# 采样点光源
light_point = bproc.types.Light()  # 创建一个光源
light_point.set_energy(120)  # 设置光强

# 加载cc_textures
assets = ["paving stones", "tiles", "wood", "bricks", "metal", "wood floor",
                               "ground", "rock", "concrete", "planks", "rocks", "gravel",
                               "asphalt", "painted metal", "marble", "carpet",
                                "metal plates", "wood siding",
                               "terrazzo", "painted wood",
                                "cardboard",
                               "diamond plate", "ice", "moss",
                               "chipboard", "sponge", "tactile paving", "paper", "cork",
                               "wood chips"]
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path, used_assets=assets)

# 定义一个采样6-DoF姿势的函数，用于下面的对象姿势采样
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())  # set_rotation_euler()以欧拉角设置实体的旋转

# 激活无抗锯齿的深度渲染，并设置颜色渲染的样本数量
bproc.renderer.enable_depth_output(activate_antialiasing=False)  # 深度图

for i in range(args.num_scenes):
    
    # 对场景中的bop对象进行采样
    sampled_target_bop_objs = list(np.random.choice(target_bop_objs, size=20, replace=True))  # np.random.choice()从一维数据中随机抽数据，返回指定大小size的数组，replace: True表示可以取相同数据，False表示不可以取相同数据
    print("sampled_target_bop_objs: !!!!!!!!!!!!!!!!!!!!!!!!!!!", sampled_target_bop_objs)
    sampled_distractor_bop_objs = list(np.random.choice(tless_dist_bop_objs, size=2, replace=False))
    # sampled_distractor_bop_objs += list(np.random.choice(lm_dist_bop_objs, size=2, replace=False))

    # 随机化材料和设置物理参数
    for obj in (sampled_target_bop_objs + sampled_distractor_bop_objs):
        mat = obj.get_materials()[0]
        if obj.get_cp("bop_dataset_name") in ['tless']:  # 使用grey_col = np.random.uniform(0.1, 0.9)函数为T-LESS对象的材质采样灰色
            grey_col = np.random.uniform(0.1, 0.9)
            mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
        obj.enable_rigidbody(True, mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)
        obj.hide(False)

    # 对两个光源进行采样
    # make_emissive()使材质发光
    light_plane_material.make_emissive(emission_strength = np.random.uniform(3, 6),  # np.random.uniform()从一个均匀分布[low, high)中随机采样，左闭右开
                                                                                emission_color = np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))
    light_plane.replace_materials(light_plane_material)  # 用给定新材质替换对象的所有材质
    light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
    # 从两个球体之间的体积中采样一个点。可以通过设置仰角(elevation)和方位角(azimuth)来约束球体
    # center: 两个球体共享的中心
    # radius_min: 较小球体的半径
    # radius_max: 较大球体的半径
    # elevation_min: 最小仰角[-90, 90]
    # elevation_max: 最大仰角[-90, 90]
    # azimuth_min: 最小方位角
    # azimuth_max: 最大方位角
    location = bproc.sampler.shell(center=[0, 0, 0], radius_min=1, radius_max=1.5,
                                                        elevation_min=5, elevation_max=89)
    light_point.set_location(location)  # 光源定位

    # 采样CC纹理并分配给房间平面
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)

    # 采样对象姿势并检查碰撞
    # 在执行网格和边界框碰撞检查时，在采样体积内对选定对象的位置和旋转进行采样
    # objects_to_sample: 基于给定函数对其姿势进行采样的网格对象列表
    # sample_pose_func: 用于对给定对象的姿势进行采样的函数
    # max_tries: 最大尝次数
    bproc.object.sample_poses(objects_to_sample=sampled_target_bop_objs + sampled_distractor_bop_objs,
                                                            sample_pose_func=sample_pose_func,
                                                            max_tries=1000)

    # 物理定位
    # 模拟当前场景并最终修复所有活动对象的最终姿势
    # 模拟运行时间最少为min_simulation_time秒，最长为max_simulation_time秒，每隔check_object_interval秒，检查最后一秒的最大对象移动是否低于给定阈值。如果是，则停止模拟。
    # 执行模拟后，删除模拟缓存，禁用刚体组件，并将活动对象的位姿设置为其在模拟中的最终位姿。
    # min_simulation_time: 模拟的最小秒数
    # max_simulation_time: 模拟的最大秒数
    # check_object_interval: 检查所有对象是否仍在移动的时间间隔。如果所有对象都停止移动，则模拟将停止。
    # substeps_per_frame: 每帧采取的模拟步骤数
    # solver_iters: 每个模拟步骤进行的约束求解器迭代次数
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                                                                                max_simulation_time=10,
                                                                                                                check_object_interval=1,
                                                                                                                substeps_per_frame=20,
                                                                                                                solver_iters=25)

    # 用于相机障碍物检查的BVH树
    # 创建一个包含多个网格对象的bvh树
    # 后面用于快速光线投射
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs + sampled_distractor_bop_objs)

    cam_poses = 0
    while cam_poses < 25:
        # 相机位置采样，返回一个点，点的位置是相对于世界坐标系的
        location = bproc.sampler.shell(center=[0, 0, 0],
                                                            radius_min=0.44,
                                                            radius_max=1.42,
                                                            elevation_min=15,
                                                            elevation_max=89)
        # 将场景中的兴趣点确定为最接近一个物体子集的平均值的物体
        # point of interest
        # 计算场景中的一个兴趣点，该点被定义为所选物体中与所选物体的bboxes的平均位置最接近的一个位置
        poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, size=10, replace=False))
        print("poi: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", poi)
        
        # 根据从位置到poi的矢量来计算旋转矩阵
        # rotation_from_forward_vec(forward_vec, up_axis='Y', inplane_rot=None)
        # forward_vec: 指向相机应该看的方向的前向向量
        # up_axis: 上轴，通常是Y
        # inplane_rot: 以弧度为单位的平面内旋转。如果没有给出，平面内旋转仅基于向上向量确定
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159))

        # 根据位置和旋转添加相机姿势，相机坐标系到世界坐标系的变换矩阵
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

        # 检查障碍物离相机至少有0.3米远，确保感兴趣的视野足够
        # 根据给定的proximity_checks，检查相机前面是否有太远或太近的障碍
        # cam2world_matrix: 从相机空间到世界空间的变换矩阵
        # proximity_checks: 一个包含运算符（如avg，min）的字典作为键，包含阈值的字典作为值，形式为{"min": 1.0, "max": 4.0}的形式，或者在最大或最小的情况下只包含数字阈值。
        # bvh_tree: 包含所有应在此处有所考虑的对象的bvh树
        # sqrt_number_of_rays: 将用于确定可见对象的射线数的平方根
        # 返回值：如果给定的相机姿态不违反任何指定的proximity_checks，则为真
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            # 将新的相机姿势设置到新的或现有的帧
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    # 渲染整个管道
    data = bproc.renderer.render()

    # blenderproc.writer.write_bop(output_dir, target_objects=None, depths=None, colors=None, color_file_format='PNG', dataset='', append_to_existing_output=True, 
    #                                                                depth_scale=1.0, jpg_quality=95, save_world2cam=True, ignore_dist_thres=100.0, m2mm=True, frames_per_chunk=1000)
    # output_dir: 输出目录的路径
    # target_objects: 以 BOP 格式保存真实姿势的对象。默认值：保存所有对象或来自指定数据集
    # depths: 要保存的以 m 为单位的深度图像列表
    # colors: 要保存的彩色图像列表
    # color_file_format: 保存彩色图像的文件类型。可用：“PNG”、“JPEG”
    # jpg_quality: 如果 color_file_format 为“JPEG”，则以给定的质量保存。
    # dataset: 仅保存指定 bop 数据集对象的注释。如果未定义，则保存所有对象姿势。
    # append_to_existing_output: 如果为真，则新帧将附加到现有帧。
    # depth_scale: 将 uint16 输出深度图像与该因子相乘以获得以 mm 为单位的深度。用于在深度精度和最大深度值之间进行权衡。默认对应于 65.54m 最大深度和 1mm 精度。
    # save_world2cam: 如果为真，摄像机到世界的转换“cam_R_w2c”、“cam_t_w2c”保存在 scene_camera.json 中
    # ignore_dist_thres: 相机和物体之间的距离，之后物体被忽略。主要是因为物理失败。
    # m2mm: 原始 bop 注释和模型以 mm 为单位。如果为真，我们在这里将 gt 注释转换为 mm。如果使用 BopLoader 选项 mm2m，则需要这样做。
    # frames_per_chunk: 每个块中保存的帧数（在 BOP 中称为场景）

    # 以bop格式写入数据
    bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
                                                    target_objects=sampled_target_bop_objs,
                                                    dataset='bottle',
                                                    depth_scale=0.1,
                                                    depths=data["depth"],
                                                    colors=data["colors"],
                                                    color_file_format="JPEG",
                                                    ignore_dist_thres=10)

    for obj in (sampled_target_bop_objs + sampled_distractor_bop_objs):
        # obj.disable_rigidbody()
        obj.hide(True)








