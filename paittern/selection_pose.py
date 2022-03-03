import numpy as np
import tensorflow as tf
from tensorflow.io import read_file
from tensorflow.image import decode_gif
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#variables
nb_image_criteria_max_x = 10

#load gif and get outpût images
image_path = '../notebooks/animation.gif'
output_images = tf.io.read_file(image_path)
output_images = tf.image.decode_gif(output_images)
output_images.shape

#example for keypoints sequence
'''
keypoints_sequence = [([[[[0.13626342, 0.57170576, 0.5406235 ],
          [0.12581952, 0.58031356, 0.5950187 ],
          [0.12306002, 0.5641533 , 0.5945765 ],
          [0.1330445 , 0.58676374, 0.46940276],
          [0.1319304 , 0.55489534, 0.48606545],
          [0.19684711, 0.6048043 , 0.37372732],
          [0.19630507, 0.53479004, 0.30953246],
          [0.26008302, 0.6065161 , 0.42267993],
          [0.25835255, 0.51460385, 0.49753478],
          [0.23563783, 0.59914243, 0.28201967],
          [0.25673127, 0.49728453, 0.4280582 ],
          [0.3964715 , 0.5841227 , 0.5168887 ],
          [0.3921252 , 0.54193866, 0.47156578],
          [0.5595846 , 0.58796036, 0.54856455],
          [0.5583141 , 0.54261553, 0.52169836],
          [0.7302867 , 0.5933529 , 0.4042293 ],
          [0.7241063 , 0.5598221 , 0.30607915]]]],),
 ([[[[0.11903933, 0.56090534, 0.5169446 ],
          [0.10934742, 0.5705746 , 0.47743547],
          [0.10738903, 0.55517316, 0.4106852 ],
          [0.12880833, 0.583263  , 0.49450025],
          [0.12308775, 0.54847294, 0.5617142 ],
          [0.20026574, 0.5842665 , 0.66105044],
          [0.20587736, 0.5409353 , 0.625576  ],
          [0.2681548 , 0.55512017, 0.24498263],
          [0.28022116, 0.5021744 , 0.4482979 ],
          [0.26274723, 0.52080506, 0.3501577 ],
          [0.26232618, 0.5114571 , 0.28268808],
          [0.39501584, 0.57516474, 0.721375  ],
          [0.39688617, 0.5429188 , 0.6819158 ],
          [0.5817502 , 0.57710975, 0.6350522 ],
          [0.58042103, 0.5454107 , 0.55936444],
          [0.7393924 , 0.5918076 , 0.4752426 ],
          [0.73018193, 0.5666239 , 0.5172093 ]]]],),
 ([[[[0.13318224, 0.55952865, 0.4611247 ],
          [0.12682736, 0.56933326, 0.45320213],
          [0.12158271, 0.55364007, 0.501277  ],
          [0.14576319, 0.57971054, 0.55577385],
          [0.13637434, 0.54448086, 0.58478993],
          [0.20490585, 0.577482  , 0.599492  ],
          [0.21796092, 0.5394596 , 0.51406467],
          [0.2790457 , 0.5572089 , 0.3590548 ],
          [0.29662517, 0.514444  , 0.15053391],
          [0.25149125, 0.5236464 , 0.3911663 ],
          [0.2813504 , 0.5027687 , 0.3972476 ],
          [0.3975859 , 0.5720369 , 0.65400684],
          [0.4012463 , 0.5437833 , 0.6879749 ],
          [0.5849682 , 0.56578463, 0.5413509 ],
          [0.5883237 , 0.54231733, 0.5605563 ],
          [0.74206275, 0.59505296, 0.56969815],
          [0.7368485 , 0.56777966, 0.44001812]]]], ),
 ([[[[0.16150343, 0.5557242 , 0.51436776],
          [0.15128978, 0.5657872 , 0.6859526 ],
          [0.14789958, 0.55061036, 0.4563579 ],
          [0.16478886, 0.5766023 , 0.40980244],
          [0.15985131, 0.5429251 , 0.4547857 ],
          [0.22190446, 0.57238287, 0.6466947 ],
          [0.23967366, 0.5429239 , 0.58585244],
          [0.32350892, 0.5593353 , 0.3441053 ],
          [0.3342293 , 0.52543795, 0.47360498],
          [0.35073382, 0.5183195 , 0.18448141],
          [0.3394901 , 0.50305605, 0.37926617],
          [0.41807944, 0.57131207, 0.66985357],
          [0.42099717, 0.54650706, 0.6205033 ],
          [0.60089064, 0.5571274 , 0.64155686],
          [0.6012659 , 0.53870517, 0.52040267],
          [0.75622016, 0.5944865 , 0.56625926],
          [0.7513477 , 0.5688376 , 0.4533968 ]]]], ),
 ([[[[0.17119896, 0.55431795, 0.65762174],
          [0.1641411 , 0.5632325 , 0.5738114 ],
          [0.16227084, 0.5482719 , 0.6999848 ],
          [0.17950638, 0.574966  , 0.570612  ],
          [0.17202213, 0.54290915, 0.57458735],
          [0.23494339, 0.56739473, 0.415179  ],
          [0.24977106, 0.55462873, 0.45368198],
          [0.3402242 , 0.5621322 , 0.44017628],
          [0.3311993 , 0.5404615 , 0.28933096],
          [0.4339265 , 0.5579535 , 0.62782705],
          [0.29925185, 0.50375867, 0.4816786 ],
          [0.43029714, 0.57282275, 0.57726043],
          [0.43196964, 0.5522737 , 0.5526497 ],
          [0.60913986, 0.55608445, 0.68798673],
          [0.60523015, 0.5404867 , 0.48570338],
          [0.75797564, 0.5934522 , 0.49470276],
          [0.7497481 , 0.56782985, 0.51544404]]]], ),
 ([[[[0.16423771, 0.5542055 , 0.4476894 ],
          [0.15339926, 0.5630103 , 0.31767184],
          [0.1527423 , 0.5487698 , 0.47092825],
          [0.17066416, 0.57537574, 0.4663293 ],
          [0.16541897, 0.5459804 , 0.3224501 ],
          [0.23573993, 0.5744615 , 0.5441735 ],
          [0.2455257 , 0.5491906 , 0.4640396 ],
          [0.32704136, 0.5731228 , 0.40050858],
          [0.28452873, 0.5115518 , 0.4741663 ],
          [0.41764566, 0.55487293, 0.5904827 ],
          [0.18517944, 0.5185043 , 0.44961128],
          [0.42733592, 0.57031035, 0.6338501 ],
          [0.425437  , 0.5447785 , 0.5191246 ],
          [0.6039492 , 0.5641505 , 0.7619283 ],
          [0.6014823 , 0.54008365, 0.4261435 ],
          [0.7570421 , 0.5964624 , 0.6766449 ],
          [0.75670356, 0.5698934 , 0.42817563]]]], ),
 ([[[[0.1561889 , 0.5577465 , 0.5140868 ],
          [0.14730869, 0.5655052 , 0.37054005],
          [0.1452039 , 0.55270696, 0.43987352],
          [0.16712007, 0.57791686, 0.53598285],
          [0.1598628 , 0.54657465, 0.49885616],
          [0.23270805, 0.576758  , 0.6036379 ],
          [0.22820191, 0.5345039 , 0.42500427],
          [0.32491335, 0.5780752 , 0.3631065 ],
          [0.18163733, 0.50404596, 0.6082583 ],
          [0.42669922, 0.5803493 , 0.39094582],
          [0.09509913, 0.5176658 , 0.32440656],
          [0.41497505, 0.56788844, 0.72405165],
          [0.41109744, 0.54016644, 0.60939443],
          [0.5873271 , 0.57379884, 0.5759337 ],
          [0.59004223, 0.5462224 , 0.55275464],
          [0.7456837 , 0.597964  , 0.4772279 ],
          [0.7474223 , 0.56999147, 0.4754924 ]]]], ),
 ([[[[0.15527277, 0.5622794 , 0.49794298],
          [0.14656745, 0.569406  , 0.3414945 ],
          [0.14363106, 0.55681837, 0.6396861 ],
          [0.16225216, 0.57823366, 0.51023227],
          [0.158239  , 0.54866296, 0.55709445],
          [0.22526579, 0.5755907 , 0.6259518 ],
          [0.21736011, 0.5377183 , 0.57576555],
          [0.32049787, 0.5813277 , 0.43776307],
          [0.1792584 , 0.50437266, 0.6653438 ],
          [0.41863033, 0.5864122 , 0.29854432],
          [0.09340116, 0.5236304 , 0.4197374 ],
          [0.405904  , 0.56475085, 0.5622955 ],
          [0.40711507, 0.5373896 , 0.69289887],
          [0.58626276, 0.57290477, 0.6649076 ],
          [0.5902896 , 0.5528421 , 0.6053017 ],
          [0.7466216 , 0.5998106 , 0.45431122],
          [0.74678266, 0.57101756, 0.5631762 ]]]], ),
 ([[[[0.15711987, 0.5546105 , 0.46834493],
          [0.14813818, 0.56402886, 0.4936217 ],
          [0.145242  , 0.55080783, 0.51010567],
          [0.16704345, 0.5750708 , 0.6057619 ],
          [0.15594982, 0.54550755, 0.48390898],
          [0.22721764, 0.57266486, 0.6526016 ],
          [0.22092785, 0.5396174 , 0.34992316],
          [0.32314238, 0.5728216 , 0.23015448],
          [0.24968536, 0.50758183, 0.6646149 ],
          [0.4283996 , 0.5701141 , 0.48671868],
          [0.14992397, 0.52324796, 0.47332582],
          [0.403735  , 0.5711661 , 0.65971184],
          [0.4052191 , 0.54256606, 0.57496905],
          [0.58857816, 0.56903636, 0.6720901 ],
          [0.58978164, 0.5497353 , 0.6663368 ],
          [0.74304104, 0.5980403 , 0.47722012],
          [0.742901  , 0.5723851 , 0.4879126 ]]]], ),
 ([[[[0.16144413, 0.5525964 , 0.6227006 ],
          [0.1521914 , 0.5617894 , 0.4664936 ],
          [0.15067948, 0.54655606, 0.4608607 ],
          [0.16809882, 0.5735758 , 0.37281936],
          [0.15985614, 0.5417378 , 0.61895406],
          [0.23733275, 0.5706433 , 0.5837774 ],
          [0.23506594, 0.54418397, 0.48183823],
          [0.3477551 , 0.5614986 , 0.4755748 ],
          [0.30985045, 0.5292032 , 0.3444612 ],
          [0.41318056, 0.5509114 , 0.46152395],
          [0.251517  , 0.5225608 , 0.34234792],
          [0.41293532, 0.56841433, 0.6612867 ],
          [0.41679466, 0.5413994 , 0.6060846 ],
          [0.59029114, 0.56087506, 0.66261923],
          [0.5973724 , 0.5352674 , 0.5374586 ],
          [0.7266011 , 0.60085547, 0.5608757 ],
          [0.73613834, 0.5702447 , 0.4633465 ]]]], ),
 ([[[[0.18063694, 0.55181545, 0.7499719 ],
          [0.16933304, 0.56089914, 0.74194473],
          [0.17045228, 0.5441235 , 0.6611475 ],
          [0.18369122, 0.5753003 , 0.6328931 ],
          [0.18174891, 0.5374814 , 0.4058416 ],
          [0.2600348 , 0.5738006 , 0.6078224 ],
          [0.2597067 , 0.53001964, 0.5068872 ],
          [0.3338906 , 0.5629897 , 0.2953232 ],
          [0.3494097 , 0.5248709 , 0.4234113 ],
          [0.39645386, 0.54584825, 0.38752472],
          [0.39215675, 0.5315141 , 0.27294835],
          [0.42140308, 0.56941944, 0.6887681 ],
          [0.42741552, 0.54007137, 0.6440234 ],
          [0.59177196, 0.56569016, 0.6486347 ],
          [0.6057584 , 0.5338784 , 0.5764419 ],
          [0.7428736 , 0.59395635, 0.33918577],
          [0.743737  , 0.5664912 , 0.36343077]]]], ),
 ([[[[0.18960133, 0.5564888 , 0.6086246 ],
          [0.17758906, 0.56594664, 0.6211061 ],
          [0.17580482, 0.5495705 , 0.75438386],
          [0.18901066, 0.57878464, 0.58471763],
          [0.18649511, 0.53967863, 0.42734516],
          [0.264269  , 0.58589005, 0.6950009 ],
          [0.2673465 , 0.528035  , 0.5673595 ],
          [0.3525602 , 0.5600669 , 0.2552578 ],
          [0.3606965 , 0.52125466, 0.5789044 ],
          [0.43410575, 0.5413483 , 0.05414885],
          [0.450356  , 0.505612  , 0.3735828 ],
          [0.4272992 , 0.56935775, 0.57134014],
          [0.43233737, 0.53799516, 0.6688087 ],
          [0.59629124, 0.5693261 , 0.63956696],
          [0.6081675 , 0.5279186 , 0.6709977 ],
          [0.75592244, 0.589686  , 0.4896568 ],
          [0.7503327 , 0.5662526 , 0.41073662]]]], ),
 ([[[[0.177753  , 0.5584909 , 0.80547243],
          [0.16656582, 0.56764483, 0.7409905 ],
          [0.16577417, 0.5518421 , 0.736644  ],
          [0.17683549, 0.580188  , 0.73401743],
          [0.17403181, 0.5433063 , 0.66166174],
          [0.23896877, 0.5953956 , 0.8600792 ],
          [0.25011167, 0.52772486, 0.67167115],
          [0.2755928 , 0.5894132 , 0.57313275],
          [0.3458247 , 0.50029624, 0.8509002 ],
          [0.24292488, 0.5634545 , 0.34819877],
          [0.4250811 , 0.47019783, 0.65240985],
          [0.4253968 , 0.56996673, 0.6598244 ],
          [0.42359877, 0.5354469 , 0.700287  ],
          [0.594759  , 0.5685637 , 0.5436518 ],
          [0.5984065 , 0.53061396, 0.66415405],
          [0.75831294, 0.5858599 , 0.7203349 ],
          [0.7441281 , 0.56281996, 0.41185904]]]],),
 ([[[[0.1617902 , 0.560968  , 0.6411674 ],
          [0.14993948, 0.56990874, 0.648301  ],
          [0.14944886, 0.553074  , 0.5038237 ],
          [0.16337562, 0.58235586, 0.6052311 ],
          [0.16179426, 0.54411083, 0.49910888],
          [0.21758686, 0.5980982 , 0.66229147],
          [0.24116696, 0.5359291 , 0.63486207],
          [0.25925392, 0.5825379 , 0.32886803],
          [0.29488957, 0.48767877, 0.70624423],
          [0.23823589, 0.5549628 , 0.51011354],
          [0.33212888, 0.44364834, 0.50609356],
          [0.4174843 , 0.5731703 , 0.8313545 ],
          [0.41580856, 0.5378862 , 0.85882545],
          [0.5924825 , 0.5705304 , 0.65100473],
          [0.59244114, 0.5360128 , 0.8362175 ],
          [0.7595209 , 0.5854957 , 0.7710954 ],
          [0.7440269 , 0.56514126, 0.49699098]]]],),
 ([[[[0.14948791, 0.563563  , 0.6363115 ],
          [0.138212  , 0.57439476, 0.43699557],
          [0.13512857, 0.55740523, 0.62758523],
          [0.1550819 , 0.58674085, 0.55699646],
          [0.14677915, 0.5486299 , 0.4594719 ],
          [0.22547717, 0.59059143, 0.58938503],
          [0.22605556, 0.5360521 , 0.5158414 ],
          [0.28460246, 0.5394135 , 0.42189062],
          [0.23594883, 0.49465367, 0.5555953 ],
          [0.23760968, 0.4929094 , 0.16473752],
          [0.19870512, 0.44282183, 0.5042647 ],
          [0.41003457, 0.57256454, 0.44990867],
          [0.40933168, 0.5399009 , 0.58128464],
          [0.5909013 , 0.57192993, 0.5307054 ],
          [0.5912184 , 0.53181154, 0.59421384],
          [0.7657285 , 0.58455145, 0.5269074 ],
          [0.74120796, 0.5628308 , 0.35433954]]]],),
 ([[[[0.173458  , 0.5734829 , 0.57009995],
          [0.16438065, 0.58341634, 0.49184796],
          [0.15730196, 0.5680753 , 0.7016344 ],
          [0.17977943, 0.5926206 , 0.41131425],
          [0.16395013, 0.55852973, 0.4600526 ],
          [0.24837591, 0.5809449 , 0.34483904],
          [0.22799581, 0.5464623 , 0.41084674],
          [0.32744676, 0.5491841 , 0.33238736],
          [0.2304503 , 0.5181733 , 0.54682136],
          [0.3645178 , 0.50764805, 0.30267608],
          [0.15379107, 0.5101603 , 0.5365043 ],
          [0.42144215, 0.573926  , 0.40450665],
          [0.42739546, 0.5459261 , 0.532373  ],
          [0.60379976, 0.5802181 , 0.57733977],
          [0.61808664, 0.52935505, 0.70227563],
          [0.7771898 , 0.59001136, 0.27693594],
          [0.7511566 , 0.571055  , 0.33524433]]]],),
 ([[[[0.18784682, 0.5780539 , 0.5757012 ],
          [0.17905428, 0.5898549 , 0.49806416],
          [0.17245178, 0.5738153 , 0.7348342 ],
          [0.19553018, 0.598782  , 0.39553577],
          [0.1769423 , 0.56533206, 0.42207968],
          [0.26972765, 0.5900394 , 0.5837052 ],
          [0.24890228, 0.5544698 , 0.43313056],
          [0.33720094, 0.5537386 , 0.32005528],
          [0.2613881 , 0.5259602 , 0.5417355 ],
          [0.3761323 , 0.51752186, 0.2860635 ],
          [0.20696612, 0.51757866, 0.40475494],
          [0.4241491 , 0.5793466 , 0.601391  ],
          [0.42857793, 0.54851335, 0.61859065],
          [0.6138678 , 0.5791887 , 0.60665804],
          [0.629594  , 0.5342827 , 0.6952189 ],
          [0.784904  , 0.58995533, 0.46534723],
          [0.75260043, 0.57746637, 0.51712346]]]],),
 ([[[[0.1780897 , 0.5864153 , 0.55881476],
          [0.16814443, 0.59505135, 0.45641732],
          [0.16249938, 0.5793156 , 0.74480194],
          [0.18342243, 0.604443  , 0.43412155],
          [0.17255269, 0.56946987, 0.5680225 ],
          [0.25969243, 0.5930008 , 0.58216435],
          [0.24062389, 0.5589212 , 0.543803  ],
          [0.33194196, 0.55461174, 0.35963202],
          [0.23743807, 0.5275776 , 0.56677437],
          [0.35495102, 0.51664776, 0.30940163],
          [0.18135273, 0.51651603, 0.51345724],
          [0.42721277, 0.58252907, 0.5956677 ],
          [0.42790824, 0.5503979 , 0.5809963 ],
          [0.61219645, 0.57750213, 0.60569227],
          [0.6264478 , 0.5336612 , 0.64045966],
          [0.7902674 , 0.5924292 , 0.56928456],
          [0.7537538 , 0.5737814 , 0.39341152]]]],),
 ([[[[0.17591162, 0.586202  , 0.42394876],
          [0.16420428, 0.5955659 , 0.5212367 ],
          [0.15725686, 0.5806041 , 0.46583045],
          [0.18079008, 0.604605  , 0.5404721 ],
          [0.16476347, 0.5705714 , 0.5929568 ],
          [0.25419977, 0.59192634, 0.47642738],
          [0.2313208 , 0.5601673 , 0.5003044 ],
          [0.3237864 , 0.5581118 , 0.3406562 ],
          [0.22751325, 0.53779525, 0.42259   ],
          [0.35398042, 0.51051676, 0.35152215],
          [0.18485153, 0.52596855, 0.50582236],
          [0.424154  , 0.58166456, 0.59532356],
          [0.42451066, 0.5511713 , 0.5389569 ],
          [0.6110461 , 0.5756787 , 0.4310734 ],
          [0.6206443 , 0.5328267 , 0.72932184],
          [0.77897125, 0.59053123, 0.37837034],
          [0.74719906, 0.570999  , 0.41976565]]]],),
 ([[[[0.16811666, 0.5880622 , 0.6567577 ],
          [0.15681273, 0.5984693 , 0.38648003],
          [0.15197968, 0.5821403 , 0.7748995 ],
          [0.17173226, 0.60761076, 0.49138492],
          [0.15777728, 0.57126796, 0.5257438 ],
          [0.24969462, 0.59614426, 0.56458527],
          [0.2142583 , 0.55989003, 0.59787863],
          [0.32127452, 0.56400627, 0.2509424 ],
          [0.19684033, 0.5344585 , 0.39059404],
          [0.3551372 , 0.5203828 , 0.25814915],
          [0.18614146, 0.5053324 , 0.5953007 ],
          [0.41767484, 0.58599395, 0.480126  ],
          [0.41995236, 0.5540391 , 0.532646  ],
          [0.60958713, 0.5816388 , 0.67182046],
          [0.62107784, 0.53752834, 0.6650719 ],
          [0.78454816, 0.59535646, 0.6094723 ],
          [0.7437698 , 0.57794106, 0.37948006]]]],),
 ([[[[0.16606471, 0.58656794, 0.61340356],
          [0.15302159, 0.5978176 , 0.7684088 ],
          [0.14717129, 0.58087796, 0.385989  ],
          [0.16497865, 0.60774416, 0.5268298 ],
          [0.15451889, 0.57251084, 0.63797593],
          [0.24743417, 0.60643756, 0.78354025],
          [0.20916346, 0.55916446, 0.7462472 ],
          [0.32478464, 0.5690994 , 0.3898187 ],
          [0.22247536, 0.52014524, 0.39017716],
          [0.38656202, 0.5412335 , 0.5291028 ],
          [0.239479  , 0.49410015, 0.4886338 ],
          [0.40863922, 0.5870666 , 0.75601876],
          [0.40948874, 0.55690503, 0.77835333],
          [0.6090861 , 0.5847185 , 0.7952491 ],
          [0.61312   , 0.54065895, 0.76243293],
          [0.7822914 , 0.5967723 , 0.5129223 ],
          [0.7418345 , 0.57597166, 0.4069271 ]]]],),
 ([[[[0.15244704, 0.5858585 , 0.67621756],
          [0.14171363, 0.5960194 , 0.65459603],
          [0.13812496, 0.5788073 , 0.72546864],
          [0.15580411, 0.6080876 , 0.6797203 ],
          [0.14645506, 0.56919163, 0.59172267],
          [0.2522736 , 0.6073481 , 0.723968  ],
          [0.2170361 , 0.5519371 , 0.6626203 ],
          [0.3364449 , 0.57670695, 0.23222563],
          [0.276181  , 0.51459575, 0.43455714],
          [0.4149795 , 0.5615643 , 0.46765926],
          [0.31977367, 0.48631686, 0.5919344 ],
          [0.4183057 , 0.59336233, 0.6768378 ],
          [0.41692266, 0.5545351 , 0.6199832 ],
          [0.6017478 , 0.5863839 , 0.6813182 ],
          [0.60135305, 0.54387015, 0.6605259 ],
          [0.7768592 , 0.5980012 , 0.44134954],
          [0.73747677, 0.5777354 , 0.3799392 ]]]],),
 ([[[[0.14207311, 0.5850123 , 0.5084864 ],
          [0.13011871, 0.5957837 , 0.37559623],
          [0.12924948, 0.5771139 , 0.393091  ],
          [0.14659849, 0.6089323 , 0.5863297 ],
          [0.14162755, 0.56793314, 0.71758854],
          [0.2423881 , 0.61333317, 0.5953956 ],
          [0.21400827, 0.55275375, 0.67425174],
          [0.34467348, 0.6004142 , 0.45735514],
          [0.29385358, 0.52141124, 0.5945342 ],
          [0.43046343, 0.5885776 , 0.51184934],
          [0.35768604, 0.49955058, 0.5886673 ],
          [0.41637596, 0.5915425 , 0.59210974],
          [0.41553164, 0.55274993, 0.637875  ],
          [0.6055042 , 0.5894466 , 0.6723586 ],
          [0.6050049 , 0.54991376, 0.6738241 ],
          [0.7754909 , 0.59810144, 0.48341572],
          [0.73875076, 0.5794819 , 0.4266673 ]]]],),
 ([[[[0.1370728 , 0.5793827 , 0.678422  ],
          [0.12640105, 0.588251  , 0.59400415],
          [0.12687814, 0.5726416 , 0.5007332 ],
          [0.13912584, 0.602839  , 0.5382752 ],
          [0.13778108, 0.565471  , 0.67769074],
          [0.23193601, 0.6136921 , 0.5497782 ],
          [0.20966095, 0.55013984, 0.48946765],
          [0.34630227, 0.61076605, 0.37226358],
          [0.30609295, 0.5312033 , 0.37175536],
          [0.4511372 , 0.60286254, 0.4042207 ],
          [0.3869284 , 0.5129058 , 0.41712162],
          [0.41093042, 0.5878522 , 0.5791489 ],
          [0.40632147, 0.55105907, 0.5921806 ],
          [0.5983084 , 0.5843528 , 0.5693281 ],
          [0.59308994, 0.5520222 , 0.6536418 ],
          [0.7763512 , 0.5991184 , 0.6913118 ],
          [0.7328783 , 0.5794494 , 0.48614925]]]],),
 ([[[[0.1433694 , 0.57769275, 0.6221306 ],
          [0.13133027, 0.5859251 , 0.61566645],
          [0.13284014, 0.5697918 , 0.42852968],
          [0.14574663, 0.60028136, 0.5102097 ],
          [0.14747086, 0.5606496 , 0.5571088 ],
          [0.23329683, 0.6146227 , 0.6420633 ],
          [0.22222121, 0.54646   , 0.62130827],
          [0.34640634, 0.62125087, 0.34641182],
          [0.33016714, 0.53036994, 0.57868195],
          [0.46046767, 0.6193052 , 0.5297183 ],
          [0.41647124, 0.5189622 , 0.56249   ],
          [0.41534695, 0.58788997, 0.62239295],
          [0.41041335, 0.5490193 , 0.60370135],
          [0.5981271 , 0.5849688 , 0.4168299 ],
          [0.5980041 , 0.55153394, 0.48633367],
          [0.7756543 , 0.5987241 , 0.6742853 ],
          [0.737094  , 0.5826165 , 0.4185012 ]]]],),
 ([[[[0.15447435, 0.57312477, 0.6362023 ],
          [0.14070687, 0.5808382 , 0.3976565 ],
          [0.14582013, 0.56411827, 0.599833  ],
          [0.15131745, 0.5942272 , 0.6996551 ],
          [0.16312803, 0.55635315, 0.6061292 ],
          [0.24533695, 0.6148348 , 0.7120513 ],
          [0.25072554, 0.54473567, 0.53342855],
          [0.35665905, 0.62034667, 0.43398255],
          [0.3639159 , 0.53128344, 0.41677126],
          [0.46504432, 0.6221597 , 0.44471142],
          [0.44833666, 0.5214092 , 0.4481358 ],
          [0.4267326 , 0.5935524 , 0.6536389 ],
          [0.42601216, 0.5510999 , 0.6597203 ],
          [0.5996934 , 0.585542  , 0.45434812],
          [0.60474974, 0.55534077, 0.5134163 ],
          [0.76835376, 0.5872397 , 0.3761023 ],
          [0.7443679 , 0.58428586, 0.56812096]]]],),
 ([[[[0.1741699 , 0.5684536 , 0.56520456],
          [0.15958822, 0.57511216, 0.545453  ],
          [0.16386046, 0.5597309 , 0.7330948 ],
          [0.16697976, 0.5885692 , 0.5611712 ],
          [0.17883594, 0.5542869 , 0.53598017],
          [0.2498739 , 0.61106396, 0.6130216 ],
          [0.26368934, 0.5463596 , 0.71297854],
          [0.3625163 , 0.61940485, 0.49271125],
          [0.3661102 , 0.5357971 , 0.4814872 ],
          [0.4697088 , 0.6256584 , 0.5775584 ],
          [0.45797998, 0.52158916, 0.49383417],
          [0.43917218, 0.59269387, 0.6822581 ],
          [0.4371376 , 0.55095744, 0.559707  ],
          [0.6037097 , 0.58787125, 0.6799346 ],
          [0.60750055, 0.5563293 , 0.6514703 ],
          [0.766854  , 0.58729124, 0.34740436],
          [0.7516203 , 0.5848697 , 0.60795903]]]],),
 ([[[[0.17943232, 0.566152  , 0.6118429 ],
          [0.16546038, 0.5738163 , 0.5740548 ],
          [0.16885939, 0.55864227, 0.7272292 ],
          [0.17487548, 0.5881144 , 0.3882029 ],
          [0.18198176, 0.5540738 , 0.43709219],
          [0.25469145, 0.6101208 , 0.5980996 ],
          [0.2667188 , 0.54629976, 0.6911665 ],
          [0.3711548 , 0.6188512 , 0.40624565],
          [0.3776536 , 0.5382056 , 0.5786194 ],
          [0.48407933, 0.6255681 , 0.6220437 ],
          [0.44679007, 0.52222645, 0.36703163],
          [0.4384725 , 0.59817487, 0.70792556],
          [0.4380265 , 0.5558106 , 0.6119404 ],
          [0.6027359 , 0.5914288 , 0.60299736],
          [0.60732466, 0.55870014, 0.55525523],
          [0.75764865, 0.58977306, 0.4005556 ],
          [0.7517389 , 0.5890298 , 0.48207694]]]],),
 ([[[[0.16915458, 0.57029223, 0.5107728 ],
          [0.15792665, 0.5767157 , 0.47325704],
          [0.15955454, 0.56303084, 0.763155  ],
          [0.16979589, 0.58914137, 0.59193385],
          [0.17406337, 0.55688363, 0.67071706],
          [0.24925195, 0.6093961 , 0.5194118 ],
          [0.25890192, 0.54587793, 0.6913648 ],
          [0.3572156 , 0.6186513 , 0.321525  ],
          [0.35522377, 0.53906715, 0.43133125],
          [0.45888877, 0.6218663 , 0.45325807],
          [0.4017363 , 0.53197455, 0.49456927],
          [0.4292881 , 0.602396  , 0.7145215 ],
          [0.428918  , 0.5575951 , 0.582738  ],
          [0.58067906, 0.5978412 , 0.5775127 ],
          [0.58776903, 0.5643501 , 0.50164443],
          [0.73499465, 0.599508  , 0.5240999 ],
          [0.7332833 , 0.5859831 , 0.52637094]]]],),
 ([[[[0.1623584 , 0.5703421 , 0.6594419 ],
          [0.15327771, 0.57681376, 0.5234269 ],
          [0.15337953, 0.5625024 , 0.28226447],
          [0.16664292, 0.58785874, 0.4645602 ],
          [0.17022024, 0.5542707 , 0.47607458],
          [0.24499348, 0.6064313 , 0.55474764],
          [0.25953376, 0.54472995, 0.63505423],
          [0.3498963 , 0.61991024, 0.39937794],
          [0.3710342 , 0.53955925, 0.58518785],
          [0.44961905, 0.62270725, 0.38102117],
          [0.33961183, 0.54984903, 0.27964386],
          [0.42119315, 0.6016184 , 0.7284039 ],
          [0.42234966, 0.5603894 , 0.671439  ],
          [0.58084965, 0.5990049 , 0.5745515 ],
          [0.5883203 , 0.56696814, 0.48774448],
          [0.73639965, 0.6035344 , 0.56170064],
          [0.736986  , 0.58924747, 0.30252445]]]],),
 ([[[[0.17349216, 0.56427723, 0.5236098 ],
          [0.16178887, 0.57046986, 0.52236605],
          [0.16322146, 0.55656576, 0.6874031 ],
          [0.17638007, 0.5828533 , 0.72497916],
          [0.17946883, 0.55041116, 0.76183915],
          [0.24724907, 0.60664934, 0.6900308 ],
          [0.2661889 , 0.5356395 , 0.7871611 ],
          [0.3592037 , 0.61628497, 0.5966688 ],
          [0.3696247 , 0.549422  , 0.5074349 ],
          [0.4646902 , 0.6258748 , 0.59548384],
          [0.29332668, 0.5630101 , 0.47110534],
          [0.42144495, 0.6031153 , 0.84182334],
          [0.42856565, 0.559704  , 0.79291236],
          [0.5857684 , 0.6084723 , 0.7321706 ],
          [0.5991622 , 0.5660539 , 0.78673285],
          [0.72647244, 0.61776626, 0.7692236 ],
          [0.7399327 , 0.59163785, 0.62649405]]]],),
 ([[[[0.1678964 , 0.56739086, 0.737422  ],
          [0.1579623 , 0.57595515, 0.61574715],
          [0.15852714, 0.5595653 , 0.49703622],
          [0.17428361, 0.58488625, 0.68662196],
          [0.1752324 , 0.5527936 , 0.67403495],
          [0.24580316, 0.60445833, 0.80161214],
          [0.24709529, 0.5370247 , 0.7794154 ],
          [0.35206273, 0.61577404, 0.7025938 ],
          [0.33646014, 0.5534773 , 0.4382422 ],
          [0.45941612, 0.6304477 , 0.4522038 ],
          [0.27124876, 0.5886658 , 0.4667045 ],
          [0.42161036, 0.6027713 , 0.8527936 ],
          [0.4250904 , 0.55953956, 0.81418824],
          [0.58231074, 0.61203706, 0.7543328 ],
          [0.5955353 , 0.57254666, 0.7320423 ],
          [0.71627486, 0.62390184, 0.77990663],
          [0.7377879 , 0.5922172 , 0.6594262 ]]]],),
 ([[[[0.16480438, 0.574511  , 0.734117  ],
          [0.15420607, 0.5827814 , 0.38773483],
          [0.15490745, 0.56731266, 0.6632515 ],
          [0.17181937, 0.5926228 , 0.5742292 ],
          [0.17160557, 0.5580225 , 0.44627827],
          [0.24729829, 0.6058707 , 0.7012434 ],
          [0.23965076, 0.5437045 , 0.56520414],
          [0.35079756, 0.6095749 , 0.66318476],
          [0.32943946, 0.5698043 , 0.5297951 ],
          [0.44443473, 0.6268496 , 0.38554084],
          [0.27854127, 0.6192216 , 0.5809453 ],
          [0.41866517, 0.6060203 , 0.8367822 ],
          [0.4220599 , 0.5607439 , 0.7087085 ],
          [0.5818454 , 0.6158801 , 0.7008974 ],
          [0.5926302 , 0.5762768 , 0.79766726],
          [0.726439  , 0.6240065 , 0.5531116 ],
          [0.74195546, 0.5917597 , 0.6258153 ]]]],),
 ([[[[0.18142343, 0.5794078 , 0.4634016 ],
          [0.1683534 , 0.58879125, 0.6871698 ],
          [0.16575994, 0.5724354 , 0.52040815],
          [0.18283501, 0.5971177 , 0.6123041 ],
          [0.17871377, 0.5631306 , 0.46033582],
          [0.2541559 , 0.5986142 , 0.41854948],
          [0.24458273, 0.5544656 , 0.52886784],
          [0.35068423, 0.59845936, 0.37183708],
          [0.34208494, 0.58535904, 0.55646527],
          [0.38665882, 0.61769444, 0.0628657 ],
          [0.32549217, 0.6366903 , 0.5039617 ],
          [0.4266169 , 0.6040628 , 0.69521654],
          [0.4290036 , 0.5628034 , 0.5983171 ],
          [0.5862721 , 0.61466414, 0.7253437 ],
          [0.5958309 , 0.5779627 , 0.6522708 ],
          [0.7335702 , 0.62363774, 0.5540928 ],
          [0.7436339 , 0.5893491 , 0.47933048]]]],),
 ([[[[0.19455415, 0.58275664, 0.40083078],
          [0.18441735, 0.59116256, 0.5120873 ],
          [0.17931628, 0.57723665, 0.68139374],
          [0.20239854, 0.59954447, 0.5321794 ],
          [0.19131427, 0.56707126, 0.47966856],
          [0.27010915, 0.5937535 , 0.5733896 ],
          [0.2538349 , 0.55740887, 0.5235234 ],
          [0.36861897, 0.59757346, 0.38455927],
          [0.35855374, 0.5859516 , 0.48298588],
          [0.4319521 , 0.61617   , 0.05556279],
          [0.3778301 , 0.64086634, 0.4301861 ],
          [0.4367908 , 0.59965074, 0.71490014],
          [0.43923023, 0.5617117 , 0.56693524],
          [0.5870793 , 0.6089012 , 0.678718  ],
          [0.59571415, 0.57601565, 0.65675026],
          [0.7346329 , 0.6180587 , 0.57662433],
          [0.7398239 , 0.5892012 , 0.37975836]]]],),
 ([[[[0.20943914, 0.5877446 , 0.55067205],
          [0.20043749, 0.59669465, 0.5141428 ],
          [0.19526543, 0.58135813, 0.28597522],
          [0.22091909, 0.60444534, 0.5686213 ],
          [0.20652325, 0.5708196 , 0.60899943],
          [0.28977385, 0.5975401 , 0.5106929 ],
          [0.26455507, 0.56378764, 0.5606505 ],
          [0.38509178, 0.6055334 , 0.3636796 ],
          [0.38441372, 0.59743464, 0.59494174],
          [0.45869905, 0.6255547 , 0.13715494],
          [0.43811405, 0.63058364, 0.43266654],
          [0.44776464, 0.5949048 , 0.7346405 ],
          [0.4483504 , 0.55890626, 0.7816459 ],
          [0.5889276 , 0.60392845, 0.65673506],
          [0.59660894, 0.57327735, 0.6267054 ],
          [0.73057735, 0.6109189 , 0.61866456],
          [0.73620945, 0.5869001 , 0.40671292]]]],),
 ([[[[0.22370684, 0.5863048 , 0.66288185],
          [0.21429522, 0.5947762 , 0.43872315],
          [0.21492317, 0.58155644, 0.73345685],
          [0.23182723, 0.6052906 , 0.53301394],
          [0.22603433, 0.574119  , 0.74457735],
          [0.30314708, 0.60830986, 0.58687466],
          [0.27522466, 0.5626656 , 0.6577365 ],
          [0.39504856, 0.59873086, 0.35782003],
          [0.38306302, 0.5835664 , 0.45732564],
          [0.45801374, 0.62190807, 0.39568728],
          [0.4545228 , 0.61135477, 0.4586758 ],
          [0.461021  , 0.5934905 , 0.77574337],
          [0.4571302 , 0.5558855 , 0.64868546],
          [0.598958  , 0.59994423, 0.6767858 ],
          [0.6053541 , 0.5728213 , 0.61493284],
          [0.7337675 , 0.6017856 , 0.4345832 ],
          [0.73926973, 0.58553946, 0.43672693]]]],),
 ([[[[0.24638397, 0.5843548 , 0.5937711 ],
          [0.23748277, 0.5929387 , 0.66198766],
          [0.23767096, 0.5785475 , 0.7625158 ],
          [0.2514401 , 0.6053126 , 0.60126144],
          [0.24842829, 0.5722049 , 0.67861146],
          [0.32749122, 0.60951585, 0.47266585],
          [0.31055188, 0.56400466, 0.70428675],
          [0.42486376, 0.6071094 , 0.30267188],
          [0.41798368, 0.5783661 , 0.3838964 ],
          [0.4878144 , 0.6167987 , 0.27258593],
          [0.4843339 , 0.60567003, 0.40098828],
          [0.4800515 , 0.59173256, 0.6375046 ],
          [0.47260752, 0.55223644, 0.6148299 ],
          [0.5945187 , 0.59348637, 0.46276096],
          [0.5901332 , 0.5659348 , 0.4190548 ],
          [0.73303086, 0.5874138 , 0.46000025],
          [0.72425854, 0.58465046, 0.4116979 ]]]],),
 ([[[[0.2899952 , 0.5797259 , 0.50972587],
          [0.27936202, 0.58741766, 0.4728423 ],
          [0.2806519 , 0.5733396 , 0.40663767],
          [0.2887542 , 0.6001867 , 0.56363654],
          [0.29068503, 0.56757534, 0.56455153],
          [0.36465287, 0.6102201 , 0.48043236],
          [0.34953302, 0.55862397, 0.673518  ],
          [0.4513844 , 0.608536  , 0.44115856],
          [0.4513716 , 0.5665277 , 0.44578776],
          [0.52097356, 0.61259943, 0.3569489 ],
          [0.50749695, 0.59846103, 0.34404427],
          [0.51410234, 0.5895768 , 0.6072569 ],
          [0.50740576, 0.54982823, 0.5968299 ],
          [0.5985161 , 0.5861601 , 0.33057794],
          [0.5957442 , 0.5590729 , 0.24374184],
          [0.7316123 , 0.5792583 , 0.47827557],
          [0.72047925, 0.5836731 , 0.38267744]]]],),
 ([[[[0.34528354, 0.576994  , 0.30443043],
          [0.33357617, 0.58359826, 0.45633283],
          [0.3343891 , 0.5710752 , 0.64496064],
          [0.34433463, 0.594263  , 0.47039253],
          [0.34575155, 0.5631918 , 0.3580215 ],
          [0.41068706, 0.60514975, 0.4478462 ],
          [0.40593833, 0.5494599 , 0.6001179 ],
          [0.50311834, 0.60874397, 0.38450298],
          [0.50061065, 0.5457422 , 0.38212395],
          [0.5545222 , 0.6121863 , 0.35725716],
          [0.51944566, 0.5890689 , 0.31959283],
          [0.5531726 , 0.5882351 , 0.6983372 ],
          [0.5465581 , 0.5487525 , 0.5655255 ],
          [0.6446693 , 0.5809391 , 0.2476283 ],
          [0.5882686 , 0.5556463 , 0.4213704 ],
          [0.732112  , 0.57442564, 0.36079168],
          [0.72189236, 0.5801439 , 0.35109422]]]],),
 ([[[[0.33882993, 0.573843  , 0.66969836],
          [0.32901058, 0.58142513, 0.5523921 ],
          [0.33048004, 0.5671496 , 0.8052978 ],
          [0.34110466, 0.59458333, 0.42994133],
          [0.3391328 , 0.56097716, 0.6235293 ],
          [0.41800952, 0.6034169 , 0.6925292 ],
          [0.4021939 , 0.54752797, 0.7016266 ],
          [0.5003591 , 0.6067851 , 0.49356318],
          [0.48798227, 0.5471862 , 0.64689255],
          [0.5449274 , 0.6110565 , 0.41692156],
          [0.515964  , 0.58630514, 0.44046238],
          [0.5373993 , 0.5867997 , 0.6483996 ],
          [0.5358274 , 0.54706705, 0.46516478],
          [0.56996995, 0.5884948 , 0.39908692],
          [0.57060206, 0.5566301 , 0.54725945],
          [0.7085782 , 0.57924354, 0.28127348],
          [0.7204979 , 0.56791145, 0.3675    ]]]],),
 ([[[[0.34631655, 0.5732302 , 0.33948648],
          [0.33420828, 0.5804412 , 0.39675832],
          [0.3348884 , 0.56628096, 0.55670345],
          [0.34499   , 0.5914565 , 0.31100336],
          [0.3456842 , 0.5589768 , 0.4370049 ],
          [0.41522607, 0.6034828 , 0.69054335],
          [0.40364334, 0.54850054, 0.6812315 ],
          [0.49284637, 0.6084244 , 0.5137283 ],
          [0.47863507, 0.54740894, 0.6750358 ],
          [0.5061622 , 0.60646373, 0.22925842],
          [0.4865337 , 0.5837446 , 0.39707682],
          [0.5286248 , 0.5882365 , 0.6906925 ],
          [0.5272703 , 0.55133927, 0.63878703],
          [0.5568673 , 0.58865964, 0.5785126 ],
          [0.54645514, 0.5606271 , 0.40555727],
          [0.68539274, 0.5906026 , 0.4990257 ],
          [0.69646126, 0.5595926 , 0.39343724]]]],)]
'''


#reshape keypoints_sequence
def resize_matrix_keypoints(matrix_keypoints):
    return np.reshape(np.array(matrix_keypoints),(np.array(matrix_keypoints).shape[0],17,3))
keypoints_sequence_1 = resize_matrix_keypoints(keypoints_sequence)


#PICTURE FRONT
# 1st criteria calculate abs distance (on x) between 'left_shoulder', 'right_shoulder'
def distance_x_shoulders(image_serie, matrix_keypoints):
    distance_shoulder = []
    for image in matrix_keypoints:
        distance_shoulder.append(abs(image[5][1]-image[6][1]))

    # Sorting from Max to Min distance between shoulders on x
    list_shoulderx_sort = sorted(distance_shoulder, reverse=True)

    #Image index based on ranking for max(x) for shoulders
    image_index_ranking_x = []
    for i in list_shoulderx_sort:
        image_index_ranking_x.append(distance_shoulder.index(i))

    #List of n images index based on criteria 1(max(x))
    image_index_ranking_x_selection = image_index_ranking_x[ 0 : nb_image_criteria_max_x]

    #List of keypoints (after selection with criteria 1) to apply 2nd criteria
    keypoints_sequence_1_crit_1 = []
    for i in image_index_ranking_x_selection:
        keypoints_sequence_1_crit_1.append(keypoints_sequence_1[i])

    return  image_serie[image_index_ranking_x_selection[0]], matrix_keypoints[image_index_ranking_x_selection[0]], keypoints_sequence_1_crit_1

#Definition of list of keypoints (after selection with criteria 1) to apply 2nd criteria
keypoints_sequence_1_crit_1 = distance_x_shoulders(output_images, keypoints_sequence_1)[2]

#display best Picture Front with criteria 1 + keypoints matrix
plt.imshow(distance_x_shoulders(output_images, keypoints_sequence_1)[0])
distance_x_shoulders(output_images, keypoints_sequence_1)[1]

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------

#PICTURE FRONT (after applyring criteria 1)
#2nd criteria (2b) to minimize distance between y : standard deviation between 6 keypoints(shoulders,elbows,wrists)
def distance_min_y_shou_elb_wrist(image_serie, matrix_keypoints):
    horizontal_selection_2b = []
    for image in keypoints_sequence_1_crit_1:
        a  = np.std([image[5][0],image[6][0], image[7][0], image[8][0], image[9][0], image[10][0]])
        horizontal_selection_2b.append(a)

    #Min standard deviation - keypoints sequence in keypoints_sequence_1_crit_1
    min_sdt = keypoints_sequence_1_crit_1[horizontal_selection_2b.index(min(horizontal_selection_2b))]

    #Image index in keypoints_sequence_1
    for idx, arr in enumerate(keypoints_sequence_1):
        comparaison = arr == min_sdt
        if comparaison.all():
            image_index_selection_2b = idx
            break

    return image_serie[image_index_selection_2b], matrix_keypoints[image_index_selection_2b]

#display best picture front with criteria 1 + critera 2 + keypoints matrix
plt.imshow(distance_min_y_shou_elb_wrist(output_images, keypoints_sequence_1)[0]),
distance_min_y_shou_elb_wrist(output_images, keypoints_sequence_1)[1]

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------

#PICTURE PROFILE
#Criteria 3: minimize distance between x : standard deviation between 8 keypoints (shoulders/hips/knees/ankles)

def distance_min_x_shoul_hip_knee_ankle(image_serie, matrix_keypoints):
    profil_selection = []
    for image in matrix_keypoints:
        a  = np.std([image[5][1],image[6][1], image[11][1], image[12][1], image[13][1], image[14][1], image[15][1], image[16][1]] )
        profil_selection.append(a)
    profil_selection_sorted = sorted(profil_selection, reverse=False)

    #List of images index based on ranking for min(standard deviation) on x: shoulders/hips/knees/ankles
    image_index_profil_a = []
    for i in profil_selection_sorted:
        image_index_profil_a.append(profil_selection.index(i))

    return image_serie[image_index_profil_a[0]], matrix_keypoints[image_index_profil_a[0]]

#display best profile image with criteria 3 + keypoints matrix
plt.imshow(distance_min_x_shoul_hip_knee_ankle(output_images, keypoints_sequence_1)[0])
distance_min_x_shoul_hip_knee_ankle(output_images, keypoints_sequence_1)[1]
