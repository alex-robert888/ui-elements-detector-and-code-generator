DATASET_ROOT_DIRECTORY = 'C:/dataset'
LABELS_PATH = f'{DATASET_ROOT_DIRECTORY}/labels.csv'
TRAIN_DATA_PATH = f'{DATASET_ROOT_DIRECTORY}/train.csv'
TEST_DATA_PATH = f'{DATASET_ROOT_DIRECTORY}/test.csv'
CALLBACK_CHECKPOINT_PATH = r"D:\facultate\ANUL_3\sem_1\computer-vision-and-deep-learning\project\image-classifier\training_resnet50"
MODEL_SAVE_PATH = "models\\m_resnet50"


LABELS_DICT = {
  'alert': 0,
  'button': 1,
  'card': 2,
  'checkbox_checked': 3,
  'checkbox_unchecked': 4,
  'chip': 5,
  'data_table': 6,
  'dropdown_menu': 7,
  'floating_action_button': 8,
  'grid_list': 9,
  'image': 10,
  'label': 11,
  'menu': 12,
  'radio_button_checked': 13,
  'radio_button_unchecked': 14,
  'slider': 15,
  'switch_disabled': 16,
  'switch_enabled': 17,
  'text_area': 18,
  'text_field': 19,
  'tooltip': 20
}

LABELS_LIST = [
  'alert',
  'button',
  'card',
  'checkbox_checked',
  'checkbox_unchecked',
  'chip',
  'data_table',
  'dropdown_menu',
  'floating_action_button',
  'grid_list',
  'image',
  'label',
  'menu',
  'radio_button_checked',
  'radio_button_unchecked',
  'slider',
  'switch_disabled',
  'switch_enabled',
  'text_area',
  'text_field',
  'tooltip'
]

IMAGE_SIZE = 224