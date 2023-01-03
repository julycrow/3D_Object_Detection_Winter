from .votr_backbone import VoxelTransformer, VoxelTransformerV2, VoxelTransformerV3
from .my_votr_backbone import MyVoxelTransformer
from .votr_backbone_addsinpe import AddSinPEVoxelTransformer
from .votr_backbone_performer import VoxelPerformer
from .votr_backbone_local import LocalVoxelTransformer
from .votr_backbone_fast import FastVoxelTransformer
from .votr_backbone_local_clustering import LocalClusteringVoxelTransformer
from .votr_backbone_reformer import VoxelReformer
from .votr_backbone_reformerv2 import VoxelReformerV2
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x
from .votr_backbone_reformer_mlp import VoxelReformerMlp


#from .votr_new_backbone import VoxelTransformer

__all__ = {
    'VoxelTransformer': VoxelTransformer,
    'VoxelTransformerV2': VoxelTransformerV2,
    'VoxelTransformerV3': VoxelTransformerV3,
    'AddSinPEVoxelTransformer': AddSinPEVoxelTransformer,
    'MyVoxelTransformer': MyVoxelTransformer,
    'VoxelPerformer': VoxelPerformer,
    'LocalVoxelTransformer': LocalVoxelTransformer,
    'FastVoxelTransformer': FastVoxelTransformer,
    'LocalClusteringVoxelTransformer': LocalClusteringVoxelTransformer,
    'VoxelReformer': VoxelReformer,
    'VoxelReformerV2': VoxelReformerV2,
    'VoxelBackBone8x': VoxelBackBone8x,
    'VoxelReformerMlp': VoxelReformerMlp,
}
