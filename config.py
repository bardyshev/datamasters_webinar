import albumentations as A
from classification_models.tfkeras import Classifiers

SEED = 42
IDX2LABEL = {
    0: 'n02111500-Great_Pyrenees',
    1: 'n02099712-Labrador_retriever',
    2: 'n02093754-Border_terrier',
    3: 'n02096294-Australian_terrier',
    4: 'n02088632-bluetick',
    5: 'n02104365-schipperke',
    6: 'n02108422-bull_mastiff',
    7: 'n02115641-dingo',
    8: 'n02108551-Tibetan_mastiff',
    9: 'n02096437-Dandie_Dinmont',
    10: 'n02108915-French_bulldog',
    11: 'n02102177-Welsh_springer_spaniel',
    12: 'n02092002-Scottish_deerhound',
    13: 'n02099601-golden_retriever',
    14: 'n02111277-Newfoundland',
    15: 'n02091134-whippet',
}
LABEL2IDX = {v: k for k, v in IDX2LABEL.items()}
NUM_CLASSES = len(IDX2LABEL)

AVAILABLE_MODELS = Classifiers.models_names()

AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(p=0.2),
    A.RandomBrightnessContrast(p=0.3)
])

MODEL_DIR = "models"
