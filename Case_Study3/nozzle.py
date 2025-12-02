import os
import tempfile
import shutil
import time
from pathlib import Path
from ansys.geometry.core.designer import DesignFileFormat
from ansys.geometry.core import launch_modeler
from ansys.geometry.core.sketch import Sketch
from ansys.geometry.core.math import Point2D
from ansys.geometry.core.misc import UNITS
import ansys.fluent.core as pyfluent
import pandas as pd

def simulate_nozzle_and_get_max_mach(
    x1,
    x2,
    x3,
    cleanup_tmp=True,        # 运行结束后删除项目内的临时目录 (nozzle_result/tmp)
    use_project_temp=False,  # 不将临时目录重定向到项目路径，使用系统默认 TEMP
):
    """
    创建一个二维喷嘴表面模型，并将其保存为多种文件格式。

    参数:
        x1 (float): 控制壁面形状的参数。
        x2 (float): 控制壁面高度的参数。
        x3 (float): 控制出口高度的参数。
        cleanup_tmp (bool): 运行结束后是否删除项目内的临时目录 nozzle_result/tmp。
        use_project_temp (bool): 是否将临时目录重定向到项目路径；关闭时使用系统 TEMP。

    返回:
        None
    """
    tmp_dir = Path(Path(__file__).parent, "nozzle_result", "tmp") if use_project_temp else None
    if use_project_temp:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TMP"] = str(tmp_dir)
        os.environ["TEMP"] = str(tmp_dir)
        tempfile.tempdir = str(tmp_dir)
    modeler = launch_modeler()

    # 创建草图
    sketch = Sketch()
    (
        sketch.segment(start=Point2D([0, 0], unit=UNITS.m), end=Point2D([1.2, 0], unit=UNITS.m))  # axis
        .segment_to_point(end=Point2D([1.2, x3], unit=UNITS.m))  # outlet
        .segment_to_point(end=Point2D([1.2 * x1 / (x1 + 1), x2], unit=UNITS.m))  # wall
        .segment_to_point(end=Point2D([0, 0.51], unit=UNITS.m))  # wall
        .segment_to_point(end=Point2D([0, 0], unit=UNITS.m))  # inlet
    )

    # 创建设计对象
    design = modeler.create_design("Nozzle2DSurfaceDesign")

    # 使用草图创建二维表面
    surface = design.create_surface(name="Nozzle2DSurface", sketch=sketch)

    # 获取草图的边界
    edges = surface.edges

    # 定义边界的命名规则
    inlet_edges = []
    axis_edges = []
    outlet_edges = []
    wall_edges = []
    for edge in edges:
        start = edge.start
        end = edge.end

        if start.x == 0 and end.x == 0:
            inlet_edges.append(edge)
        elif start.y == 0 and end.y == 0:
            axis_edges.append(edge)
        elif start.x == end.x and start.x > 0:
            outlet_edges.append(edge)
        else:
            wall_edges.append(edge)

    print("Inlet edges:", inlet_edges)
    print("Axis edges:", axis_edges)
    print("Outlet edges:", outlet_edges)
    print("Wall edges:", wall_edges)

    # 创建命名选择
    design.create_named_selection("inlet", edges=inlet_edges)
    design.create_named_selection("axis", edges=axis_edges)
    design.create_named_selection("outlet", edges=outlet_edges)
    design.create_named_selection("wall", edges=wall_edges)
    save_dir = Path(Path(__file__).parent, "nozzle_result", f"Model_x1={x1:.3f}_x2={x2:.3f}_x3={x3:.3f}")
    save_dir.mkdir(parents=True, exist_ok=True)
    orig_cwd = Path.cwd()
    os.chdir(save_dir)
    csv_file_path = Path(save_dir, "mach_data.csv")

    scdocx_path = Path(save_dir, "Nozzle2D_Surface.scdocx")
    try:
        design.save(file_location=scdocx_path)
    except Exception:
        try:
            design.download(file_location=scdocx_path, format=DesignFileFormat.SCDOCX)
        except Exception:
            data = design._grpc_client.services.designs.download_file().get("data")
            scdocx_path.write_bytes(data)
            print("已使用备用下载方法保存几何文件。")
    # 关闭建模器
    modeler.close()

    print(f"二维表面模型已保存到: {save_dir}")

    import_file_name = scdocx_path

    try:
        # 启动 Fluent 的网格划分模式
        meshing = pyfluent.launch_fluent(
            mode="meshing",
            precision=pyfluent.Precision.DOUBLE,
            processor_count=2,
        )
        two_dim_mesh = meshing.two_dimensional_meshing()

        # 加载 CAD 几何文件
        print("加载几何文件中...")
        two_dim_mesh.load_cad_geometry_2d.file_name = str(import_file_name)
        two_dim_mesh.load_cad_geometry_2d.length_unit = "mm"
        two_dim_mesh.load_cad_geometry_2d.refaceting.refacet = False
        two_dim_mesh.load_cad_geometry_2d()

        # 更新区域和边界
        print("更新区域和边界中...")
        two_dim_mesh.update_boundaries_2d.boundary_label_list.set_state(["inlet"])
        two_dim_mesh.update_boundaries_2d.boundary_label_type_list.set_state(["wall"])
        two_dim_mesh.update_boundaries_2d.boundary_label_type_list.set_state(["outlet"])
        two_dim_mesh.update_boundaries_2d.boundary_label_type_list.set_state(["axis"])
        two_dim_mesh.update_boundaries_2d()
        two_dim_mesh.update_regions_2d()
        two_dim_mesh.update_boundaries_2d.selection_type = "zone"

        # 定义全局网格划分设置
        print("设置全局网格划分参数...")
        two_dim_mesh.define_global_sizing_2d.curvature_normal_angle = 20
        two_dim_mesh.define_global_sizing_2d.max_size = 5  # 最大网格尺寸
        two_dim_mesh.define_global_sizing_2d.min_size = 0.5  # 最小网格尺寸
        two_dim_mesh.define_global_sizing_2d.size_functions = "Curvature"
        two_dim_mesh.define_global_sizing_2d()

        # 生成表面网格
        print("生成表面网格中...")
        # two_dim_mesh.generate_initial_surface_mesh.surface2_d_preferences.merge_edge_zones_based_on_labels = "no"
        # two_dim_mesh.generate_initial_surface_mesh.surface2_d_preferences.merge_face_zones_based_on_labels = "no"
        # two_dim_mesh.generate_initial_surface_mesh.surface2_d_preferences.show_advanced_options = True
        two_dim_mesh.generate_initial_surface_mesh()

        # 检查网格质量
        print("检查网格质量中...")
        meshing.tui.mesh.check_mesh()
        # # 更新网格
        # print("更新网格中...")
        # meshing.tui.mesh.rebuild_mesh()

        # # 保存网格文件
        # print("保存网格文件...")
        # # 保存网格文件
        # mesh_file = "C:/Users/HYGK/Desktop/tqcode/tq-7-ystr/math/u-fun/population_data/laval/downloads/Nozzle2D_Surface_2d.msh.h5"  # 设置网格文件保存路径
        # meshing.tui.file.write_mesh(mesh_file)  # 使用 TUI 保存网格
        # print(f"网格划分已完成，文件保存至: {mesh_file}")
        # 设置网格文件保存路径
        mesh_file = Path(save_dir, "Nozzle2D_Surface_2d.msh.h5")

        # 获取导出网格任务
        tasks = meshing.workflow.TaskObject  # 获取所有任务
        export_mesh = tasks["Export Fluent 2D Mesh"]  # 定位导出网格任务

        # 设置导出网格参数
        export_mesh.Arguments.set_state({"FileName": str(mesh_file)})

        # 执行导出任务
        print("导出网格文件中...")
        export_mesh.Execute()
        print(f"网格文件已导出至: {mesh_file}")

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 确保安全退出 Fluent 会话

        meshing.exit()

#求解
    try:
        # **启动 Fluent 并加载网格**
        solver = pyfluent.launch_fluent(
            precision="double",
            processor_count=4,
            mode="solver",
            version="2d",
        )
        print(solver.get_fluent_version())

        # **读取网格文件**
        mesh_file_path = str(mesh_file)
        solver.file.read(file_type="mesh", file_name=mesh_file_path)
        print(f"网格文件 {mesh_file_path} 成功加载。")

        # **检查网格**
        solver.mesh.check()
        solver.tui.define.models.solver.density_based_explicit("yes")  # 启用密度基
        solver.tui.define.models.unsteady_1st_order("yes")  # 启用瞬态
        solver.setup.general.solver.two_dim_space = "axisymmetric"  # 轴对称
        # **设置模型参数**
        solver.setup.models.energy.enabled = True
        # solver.setup.models.energy = True  # 开启能量方程
        solver.setup.models.viscous = {
            "model": "k-epsilon",
        }

        # **设置材料属性**
        print("设置材料属性...")
        air = solver.setup.materials.fluid["air"]
        air.density.option = "ideal-gas"
        air.viscosity.option = "sutherland"

        # 设置 Sutherland 模型参数
        air.viscosity.sutherland.option = "three-coefficient-method"
        air.viscosity.sutherland.reference_viscosity = 1.716e-05  # kg/(m·s)
        air.viscosity.sutherland.reference_temperature = 273.11  # K
        air.viscosity.sutherland.effective_temperature = 110.56  # K

        # **设置边界条件**
        # # 设置压力入口
        print("设置边界条件...")
        # pressure_inlet = solver.setup.boundary_conditions.pressure_inlet["inlet"]
        # pressure_inlet.momentum.gauge_total_pressure = 2266675  # Pa
        # pressure_inlet.momentum.supersonic_or_initial_gauge_pressuree = 2265675  # Pa
        # pressure_inlet.thermal.temperature = 1773  # K
        # 获取压力入口的混合相设置对象
        pressure_inlet = solver.setup.boundary_conditions.pressure_inlet["inlet"]

        # 设置动量相关参数
        pressure_inlet.momentum.gauge_total_pressure = 2266675  # 总表压 (Pa)
        pressure_inlet.momentum.supersonic_or_initial_gauge_pressure = 2265675  # 超音速或初始表压 (Pa)
        pressure_inlet.momentum.direction_specification_method = "Normal to Boundary"  # 垂直边界方向

        # 设置热力学参数
        pressure_inlet.thermal.total_temperature = 1773  # 总温度 (K)

        # 设置湍流参数
        pressure_inlet.turbulence.turbulent_specification = "Intensity and Viscosity Ratio"
        pressure_inlet.turbulence.turbulent_intensity = 0.05  # 湍流强度
        pressure_inlet.turbulence.turbulent_viscosity_ratio = 10  # 湍流粘度比

        # **初始化解算器**
        print("初始化解算器...")
        solver.solution.initialization.hybrid_initialize()

        # **保存设置和网格**
        case_file_path = Path(save_dir, "2D_model.cas.h5")
        solver.file.write(file_name=str(case_file_path), file_type="case")
        print(f"设置和网格已保存至 {case_file_path}。")

        # **运行计算**
        solver.solution.run_calculation.iterate(iter_count=500)
        print("计算完成。")
        graphics = solver.results.graphics

        # 设置图像分辨率
        if graphics.picture.use_window_resolution.is_active():
            graphics.picture.use_window_resolution = False
        graphics.picture.x_resolution = 1920
        graphics.picture.y_resolution = 1440

        graphics = solver.results.graphics

        graphics.contour["contour_mach_number"] = {
            "coloring": {
                "option": "banded",
                "smooth": False,
            },
            "field": "rel-mach-number",
            "filled": True,
        }
        solver.settings.results.graphics.views.mirror_zones = ["nozzle2dsurface"]

        graphics.contour["contour_mach_number"].display()

        graphics.picture.save_picture(file_name=str(Path(save_dir, "contour_mach_number.png")))
        mach_data = solver.fields.field_data.get_scalar_field_data(
            field_name="rel-mach-number",
            surfaces=["nozzle2dsurface"]
        )
        surface_name = "nozzle2dsurface"
        mach_values = mach_data.get(surface_name)
        if mach_values is None:
            mach_values = mach_data.get(f"interior:{surface_name}")
        if mach_values is None:
            try:
                mach_values = next(iter(mach_data.values()))
            except StopIteration:
                mach_values = None

        # 检查数据是否提取成功
        if mach_values is None:
            raise ValueError(f"No data found for surface: {surface_name}")

        # 创建 DataFrame 并保存为 CSV 文件
        mach_df = pd.DataFrame({
            "Node_Index": range(1, len(mach_values) + 1),  # 创建节点索引从 1 开始
            "Mach_Number": mach_values
        })
        mach_df.to_csv(csv_file_path, index=False)
        max_value = float(pd.Series(mach_values).max())

        print("Mach number data exported successfully to 'mach_data.csv'!")

        # **保存计算结果**
        results_file_path = Path(save_dir, "2D_model_results.cas.h5")
        solver.file.write(file_name=str(results_file_path), file_type="case")
        print(f"计算结果已保存至 {results_file_path}。")


    except Exception as e:
        print(f"发生错误: {e}")
        max_value = None
    finally:
        # 确保安全退出 Fluent 会话

        solver.exit()
        os.chdir(orig_cwd)
        if cleanup_tmp and tmp_dir is not None:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    if max_value is None:
        if csv_file_path.exists():
            df = pd.read_csv(csv_file_path)
            max_value = df.iloc[:, 1].max()
        else:
            raise FileNotFoundError("未能生成马赫数数据，无法计算最大值。")

    return max_value


def cleanup_model_outputs(x1, x2, x3, delete_tmp=False):
    save_dir = Path(Path(__file__).parent, "nozzle_result", f"Model_x1{x1}_x2{x2}_x3{x3}")
    if delete_tmp:
        tmp_dir = Path(Path(__file__).parent, "nozzle_result", "tmp")
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    x1, x2, x3 = 0.6, 0.25, 0.35 #这里的参数是测试值，实际运行时可以根据需要修改
    x4, x5, x6 = 0.61, 0.26, 0.36
    max_mach= simulate_nozzle_and_get_max_mach(x1, x2, x3)
    # max_mach= simulate_nozzle_and_get_max_mach(x4, x5, x6)
    print(f"仿真完成，马赫数最大值为: {max_mach}")
