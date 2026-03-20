"""
DEM数据加载器和地形生成器
========================

功能:
1. 从GeoTIFF文件加载真实DEM数据
2. 生成模拟地形用于测试
3. 地形数据预处理和归一化
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


class DEMLoader:
    """
    DEM数据加载器
    
    支持从GeoTIFF文件加载真实DEM数据,
    并提供地形高度查询接口.
    """
    
    def __init__(
        self,
        dem_file: Optional[str] = None,
        size_x: float = 1000.0,
        size_y: float = 1000.0,
        resolution: float = 10.0,
        max_height: float = 500.0
    ):
        """
        初始化DEM加载器
        
        Args:
            dem_file: DEM文件路径 (GeoTIFF格式)
            size_x: 地形X方向尺寸 (米)
            size_y: 地形Y方向尺寸 (米)
            resolution: 网格分辨率 (米)
            max_height: 最大高程 (米)
        """
        self.dem_file = dem_file
        self.size_x = size_x
        self.size_y = size_y
        self.resolution = resolution
        self.max_height = max_height
        
        # 计算网格尺寸
        self.grid_x = int(size_x / resolution)
        self.grid_y = int(size_y / resolution)
        
        # 加载或生成地形
        if dem_file is not None:
            self.elevation_data = self._load_from_file(dem_file)
        else:
            self.elevation_data = None
            
        # 创建插值器
        self._create_interpolator()
        
    def _load_from_file(self, dem_file: str) -> np.ndarray:
        """
        从GeoTIFF文件加载DEM数据
        
        Args:
            dem_file: DEM文件路径
            
        Returns:
            高程数据数组
        """
        try:
            import rasterio
            
            with rasterio.open(dem_file) as src:
                elevation = src.read(1)
                
                # 重采样到目标分辨率
                from scipy.ndimage import zoom
                
                target_shape = (self.grid_y, self.grid_x)
                zoom_factors = (
                    target_shape[0] / elevation.shape[0],
                    target_shape[1] / elevation.shape[1]
                )
                elevation = zoom(elevation, zoom_factors, order=1)
                
                # 处理无效值
                elevation = np.nan_to_num(elevation, nan=0.0)
                
                # 归一化到指定高度范围
                elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-8)
                elevation = elevation * self.max_height
                
                return elevation
                
        except ImportError:
            print("警告: rasterio未安装，使用生成地形")
            return None
        except Exception as e:
            print(f"警告: 加载DEM文件失败 ({e})，使用生成地形")
            return None
    
    def _create_interpolator(self):
        """创建地形高度插值器"""
        if self.elevation_data is not None:
            x = np.linspace(0, self.size_x, self.grid_x)
            y = np.linspace(0, self.size_y, self.grid_y)
            self.interpolator = RegularGridInterpolator(
                (y, x), 
                self.elevation_data,
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )
        else:
            self.interpolator = None
    
    def get_elevation(self, x: float, y: float) -> float:
        """
        获取指定位置的地形高度
        
        Args:
            x: X坐标 (米)
            y: Y坐标 (米)
            
        Returns:
            地形高度 (米)
        """
        if self.interpolator is not None:
            return float(self.interpolator([[y, x]])[0])
        return 0.0
    
    def get_elevation_batch(self, positions: np.ndarray) -> np.ndarray:
        """
        批量获取地形高度
        
        Args:
            positions: 位置数组, shape (N, 2) - [[x, y], ...]
            
        Returns:
            高度数组, shape (N,)
        """
        if self.interpolator is not None:
            # 注意: RegularGridInterpolator 需要 (y, x) 顺序
            points = positions[:, [1, 0]]
            return self.interpolator(points)
        return np.zeros(len(positions))
    
    def get_elevation_grid(self) -> np.ndarray:
        """
        获取完整的地形网格数据
        
        Returns:
            高程网格, shape (grid_y, grid_x)
        """
        if self.elevation_data is not None:
            return self.elevation_data.copy()
        return np.zeros((self.grid_y, self.grid_x))
    
    def set_elevation_data(self, data: np.ndarray):
        """
        设置地形数据
        
        Args:
            data: 高程数据数组
        """
        self.elevation_data = data
        self._create_interpolator()


class TerrainGenerator:
    """
    程序化地形生成器
    
    支持生成多种类型的模拟地形用于测试和训练.
    """
    
    def __init__(
        self,
        size_x: float = 1000.0,
        size_y: float = 1000.0,
        resolution: float = 10.0,
        max_height: float = 500.0,
        seed: Optional[int] = None
    ):
        """
        初始化地形生成器
        
        Args:
            size_x: 地形X方向尺寸 (米)
            size_y: 地形Y方向尺寸 (米)
            resolution: 网格分辨率 (米)
            max_height: 最大高程 (米)
            seed: 随机种子
        """
        self.size_x = size_x
        self.size_y = size_y
        self.resolution = resolution
        self.max_height = max_height
        
        self.grid_x = int(size_x / resolution)
        self.grid_y = int(size_y / resolution)
        
        if seed is not None:
            np.random.seed(seed)
    
    def generate(self, terrain_type: str = "hills") -> DEMLoader:
        """
        生成地形
        
        Args:
            terrain_type: 地形类型
                - "flat": 平坦地形
                - "hills": 丘陵地形
                - "mountains": 山区地形
                - "valley": 峡谷地形
                - "mixed": 混合地形
                
        Returns:
            DEMLoader实例
        """
        generators = {
            "flat": self._generate_flat,
            "hills": self._generate_hills,
            "mountains": self._generate_mountains,
            "valley": self._generate_valley,
            "mixed": self._generate_mixed,
        }
        
        if terrain_type not in generators:
            raise ValueError(f"未知地形类型: {terrain_type}")
        
        elevation = generators[terrain_type]()
        
        # 创建DEMLoader并设置数据
        loader = DEMLoader(
            dem_file=None,
            size_x=self.size_x,
            size_y=self.size_y,
            resolution=self.resolution,
            max_height=self.max_height
        )
        loader.set_elevation_data(elevation)
        
        return loader
    
    def _generate_flat(self) -> np.ndarray:
        """生成平坦地形"""
        base_height = self.max_height * 0.1
        noise = np.random.randn(self.grid_y, self.grid_x) * 5
        noise = gaussian_filter(noise, sigma=3)
        return np.ones((self.grid_y, self.grid_x)) * base_height + noise
    
    def _generate_hills(self) -> np.ndarray:
        """生成丘陵地形"""
        elevation = np.zeros((self.grid_y, self.grid_x))
        
        # 添加多个高斯山丘
        n_hills = np.random.randint(5, 15)
        for _ in range(n_hills):
            cx = np.random.uniform(0, self.grid_x)
            cy = np.random.uniform(0, self.grid_y)
            height = np.random.uniform(0.3, 0.8) * self.max_height
            sigma = np.random.uniform(10, 30)
            
            x = np.arange(self.grid_x)
            y = np.arange(self.grid_y)
            xx, yy = np.meshgrid(x, y)
            
            hill = height * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
            elevation += hill
        
        # 添加噪声
        noise = np.random.randn(self.grid_y, self.grid_x) * 10
        noise = gaussian_filter(noise, sigma=2)
        elevation += noise
        
        # 裁剪到有效范围
        elevation = np.clip(elevation, 0, self.max_height)
        
        return elevation
    
    def _generate_mountains(self) -> np.ndarray:
        """生成山区地形 (使用Diamond-Square算法)"""
        # 确保尺寸是2的幂+1
        size = max(self.grid_x, self.grid_y)
        n = int(np.ceil(np.log2(size - 1)))
        ds_size = 2**n + 1
        
        elevation = self._diamond_square(ds_size, roughness=1.0)
        
        # 裁剪到目标尺寸
        elevation = elevation[:self.grid_y, :self.grid_x]
        
        # 归一化
        elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-8)
        elevation = elevation * self.max_height
        
        return elevation
    
    def _diamond_square(self, size: int, roughness: float = 1.0) -> np.ndarray:
        """Diamond-Square算法生成分形地形"""
        elevation = np.zeros((size, size))
        
        # 初始化四角
        elevation[0, 0] = np.random.random()
        elevation[0, size-1] = np.random.random()
        elevation[size-1, 0] = np.random.random()
        elevation[size-1, size-1] = np.random.random()
        
        step = size - 1
        scale = roughness
        
        while step > 1:
            half = step // 2
            
            # Diamond步骤
            for y in range(half, size - 1, step):
                for x in range(half, size - 1, step):
                    avg = (
                        elevation[y - half, x - half] +
                        elevation[y - half, x + half] +
                        elevation[y + half, x - half] +
                        elevation[y + half, x + half]
                    ) / 4.0
                    elevation[y, x] = avg + (np.random.random() - 0.5) * scale
            
            # Square步骤
            for y in range(0, size, half):
                for x in range((y + half) % step, size, step):
                    count = 0
                    total = 0.0
                    if y >= half:
                        total += elevation[y - half, x]
                        count += 1
                    if y + half < size:
                        total += elevation[y + half, x]
                        count += 1
                    if x >= half:
                        total += elevation[y, x - half]
                        count += 1
                    if x + half < size:
                        total += elevation[y, x + half]
                        count += 1
                    elevation[y, x] = total / count + (np.random.random() - 0.5) * scale
            
            step = half
            scale *= 0.5
        
        return elevation
    
    def _generate_valley(self) -> np.ndarray:
        """生成峡谷地形"""
        x = np.linspace(0, 1, self.grid_x)
        y = np.linspace(0, 1, self.grid_y)
        xx, yy = np.meshgrid(x, y)
        
        # 创建V形峡谷
        valley_center = 0.5 + 0.2 * np.sin(yy * np.pi * 2)
        valley_depth = np.abs(xx - valley_center)
        
        elevation = valley_depth * self.max_height
        
        # 添加山脊
        ridge_left = np.exp(-((xx - valley_center + 0.3)**2) / 0.02) * self.max_height * 0.5
        ridge_right = np.exp(-((xx - valley_center - 0.3)**2) / 0.02) * self.max_height * 0.5
        elevation += ridge_left + ridge_right
        
        # 添加噪声
        noise = np.random.randn(self.grid_y, self.grid_x) * 15
        noise = gaussian_filter(noise, sigma=3)
        elevation += noise
        
        elevation = np.clip(elevation, 0, self.max_height)
        
        return elevation
    
    def _generate_mixed(self) -> np.ndarray:
        """生成混合地形"""
        # 组合多种地形
        hills = self._generate_hills()
        mountains = self._generate_mountains()
        
        # 创建混合mask
        x = np.linspace(0, 1, self.grid_x)
        y = np.linspace(0, 1, self.grid_y)
        xx, yy = np.meshgrid(x, y)
        mask = (np.sin(xx * np.pi) * np.sin(yy * np.pi) + 1) / 2
        
        elevation = hills * mask + mountains * (1 - mask)
        
        return elevation


def create_terrain(config: Dict[str, Any]) -> DEMLoader:
    """
    根据配置创建地形
    
    Args:
        config: 配置字典
        
    Returns:
        DEMLoader实例
    """
    terrain_config = config['environment']['terrain']
    
    dem_file = terrain_config.get('dem_file')
    
    if dem_file is not None:
        # 从文件加载
        loader = DEMLoader(
            dem_file=dem_file,
            size_x=terrain_config['size_x'],
            size_y=terrain_config['size_y'],
            resolution=terrain_config['resolution'],
            max_height=terrain_config['max_height']
        )
    else:
        # 生成地形
        generator = TerrainGenerator(
            size_x=terrain_config['size_x'],
            size_y=terrain_config['size_y'],
            resolution=terrain_config['resolution'],
            max_height=terrain_config['max_height'],
            seed=config['training'].get('seed', 42)
        )
        loader = generator.generate(terrain_config.get('terrain_type', 'hills'))
    
    return loader


if __name__ == "__main__":
    # 测试地形生成
    import matplotlib.pyplot as plt
    
    generator = TerrainGenerator(seed=42)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    terrain_types = ["flat", "hills", "mountains", "valley", "mixed"]
    
    for ax, ttype in zip(axes.flatten(), terrain_types):
        loader = generator.generate(ttype)
        elevation = loader.get_elevation_grid()
        
        im = ax.imshow(elevation, cmap='terrain', origin='lower')
        ax.set_title(f"Terrain: {ttype}")
        plt.colorbar(im, ax=ax, label='Elevation (m)')
    
    axes[1, 2].axis('off')
    plt.tight_layout()
    plt.savefig("terrain_samples.png", dpi=150)
    plt.show()
