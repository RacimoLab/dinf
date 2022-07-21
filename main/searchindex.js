Search.setIndex({docnames:["CHANGELOG","api","cli","development","guide/abc","guide/abc-gan","guide/accuracy","guide/creating_a_model","guide/empirical_data","guide/features","guide/mcmc-gan","guide/multiple_demes","guide/performance","guide/pg-gan","guide/testing_a_model","installation","introduction"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["CHANGELOG.md","api.md","cli.md","development.md","guide/abc.ipynb","guide/abc-gan.md","guide/accuracy.ipynb","guide/creating_a_model.md","guide/empirical_data.md","guide/features.md","guide/mcmc-gan.md","guide/multiple_demes.md","guide/performance.md","guide/pg-gan.md","guide/testing_a_model.md","installation.md","introduction.md"],objects:{"dinf.BagOfVcf":[[1,1,1,"","lengths"],[1,2,1,"","sample_genotype_matrix"],[1,2,1,"","sample_regions"],[1,1,1,"","samples"]],"dinf.BinnedHaplotypeMatrix":[[1,2,1,"","from_ts"],[1,2,1,"","from_vcf"],[1,3,1,"","shape"]],"dinf.DinfModel":[[1,2,1,"","check"],[1,1,1,"","discriminator_network"],[1,1,1,"","feature_shape"],[1,1,1,"","filename"],[1,2,1,"","from_file"],[1,1,1,"","generator_func"],[1,1,1,"","generator_func_v"],[1,1,1,"","parameters"],[1,1,1,"","target_func"]],"dinf.Discriminator":[[1,2,1,"","fit"],[1,2,1,"","from_file"],[1,2,1,"","init"],[1,2,1,"","predict"],[1,2,1,"","summary"],[1,2,1,"","to_file"]],"dinf.HaplotypeMatrix":[[1,2,1,"","from_ts"],[1,2,1,"","from_vcf"],[1,3,1,"","shape"]],"dinf.MultipleBinnedHaplotypeMatrices":[[1,2,1,"","from_ts"],[1,2,1,"","from_vcf"],[1,3,1,"","shape"]],"dinf.MultipleHaplotypeMatrices":[[1,2,1,"","from_ts"],[1,2,1,"","from_vcf"],[1,3,1,"","shape"]],"dinf.Param":[[1,2,1,"","bounds_contain"],[1,2,1,"","draw_prior"],[1,1,1,"","high"],[1,2,1,"","itransform"],[1,1,1,"","low"],[1,1,1,"","name"],[1,2,1,"","reflect"],[1,2,1,"","transform"],[1,2,1,"","truncate"],[1,1,1,"","truth"]],"dinf.Parameters":[[1,2,1,"","bounds_contain"],[1,2,1,"","draw_prior"],[1,2,1,"","itransform"],[1,2,1,"","reflect"],[1,2,1,"","transform"],[1,2,1,"","truncate"]],"dinf.Store":[[1,2,1,"","__getitem__"],[1,2,1,"","__len__"],[1,2,1,"","increment"]],"dinf.plot":[[1,0,1,"","MultiPage"],[1,4,1,"","feature"],[1,4,1,"","features"],[1,4,1,"","hist"],[1,4,1,"","hist2d"],[1,4,1,"","metrics"]],dinf:[[1,0,1,"","BagOfVcf"],[1,0,1,"","BinnedHaplotypeMatrix"],[1,0,1,"","DinfModel"],[1,0,1,"","Discriminator"],[1,0,1,"","ExchangeableCNN"],[1,0,1,"","ExchangeablePGGAN"],[1,0,1,"","HaplotypeMatrix"],[1,0,1,"","MultipleBinnedHaplotypeMatrices"],[1,0,1,"","MultipleHaplotypeMatrices"],[1,0,1,"","Param"],[1,0,1,"","Parameters"],[1,0,1,"","Store"],[1,0,1,"","Symmetric"],[1,4,1,"","abc_gan"],[1,4,1,"","alfi_mcmc_gan"],[1,4,1,"","get_contig_lengths"],[1,4,1,"","get_samples_from_1kgp_metadata"],[1,4,1,"","load_results"],[1,4,1,"","mcmc_gan"],[1,4,1,"","pg_gan"],[1,5,0,"-","plot"],[1,4,1,"","predict"],[1,4,1,"","sample_smooth"],[1,4,1,"","save_results"],[1,4,1,"","train"],[1,4,1,"","ts_individuals"]]},objnames:{"0":["py","class","Python class"],"1":["py","attribute","Python attribute"],"2":["py","method","Python method"],"3":["py","property","Python property"],"4":["py","function","Python function"],"5":["py","module","Python module"]},objtypes:{"0":"py:class","1":"py:attribute","2":"py:method","3":"py:property","4":"py:function","5":"py:module"},terms:{"0":[1,4,6,7,8,9,11,15],"000":[1,4,6],"0000":6,"0022":6,"0025":6,"0026":6,"0028":6,"0030":6,"0031":6,"0036":6,"0037":6,"0038":6,"0039":6,"0040":6,"0041":6,"0042":6,"0044":6,"0045":6,"0048":6,"0050":6,"0051":6,"0052":6,"0053":6,"0054":6,"0055":6,"0056":6,"0061":6,"0063":6,"0064":6,"0065":6,"0066":6,"0067":6,"0068":6,"0070":6,"0072":6,"0074":6,"0076":6,"0077":6,"0079":6,"008":4,"0080":6,"0081":6,"0083":6,"0084":6,"0086":6,"0089":6,"0092":6,"0093":6,"0094":6,"0095":6,"0096":6,"0097":6,"0098":6,"0099":6,"0101":6,"0102":6,"0103":6,"0104":6,"0108":6,"0110":6,"0111":6,"0112":6,"0113":6,"0115":6,"0116":6,"0118":6,"0120":6,"0121":6,"0123":6,"0127":6,"0130":6,"0131":6,"0132":[4,6],"0133":6,"0136":[4,6],"0137":6,"0138":6,"0139":6,"0141":6,"0142":4,"0143":6,"0144":[4,6],"0145":6,"0146":6,"0147":4,"0151":6,"0152":6,"0153":6,"0154":4,"0155":6,"0157":6,"0159":[4,6],"0160":6,"0163":6,"0164":4,"0165":6,"0166":6,"0167":[4,6],"0168":6,"0170":6,"0174":6,"0175":4,"0177":6,"0178":6,"0179":6,"0180":4,"0181":6,"0184":6,"0187":6,"0188":[4,6],"0189":6,"0190":6,"0191":6,"0192":6,"0193":6,"0195":4,"0197":6,"0198":6,"0199":6,"02":9,"0200":6,"0204":6,"0205":4,"0207":6,"0210":6,"0211":6,"0213":6,"0214":6,"0215":6,"0216":4,"0219":6,"0221":6,"0223":6,"0224":6,"0226":6,"0228":6,"0230":6,"0231":[4,6],"0232":6,"0234":6,"02350065":4,"0237":6,"0238":6,"0239":6,"0240":6,"0241":6,"0243":4,"0244":6,"0245":6,"0246":[4,6],"0247":4,"0248":6,"025":4,"0250":6,"0251":6,"0252":[4,6],"0253":[4,6],"0254":6,"0255":6,"0258":6,"0259":6,"0260":6,"0261":4,"0262":6,"0264":6,"0267":[4,6],"0268":6,"0269":6,"0270":[4,6],"0271":6,"0272":6,"0274":6,"0276":[4,6],"0277":[4,6],"0280":6,"0281":6,"0282":6,"0283":6,"0284":6,"0285":4,"0286":6,"0287":[4,6],"0289":6,"0290":[4,6],"0291":6,"0292":6,"0293":6,"0294":[4,6],"0296":[4,6],"0297":6,"0299":6,"03":4,"0301":6,"0302":6,"0303":6,"0304":4,"0305":6,"0306":6,"0307":6,"0308":6,"0309":6,"0310":6,"0311":6,"0312":6,"0314":6,"0316":6,"0317":6,"0319":6,"0321":6,"0322":6,"0324":6,"0325":6,"0326":6,"0327":6,"0328":6,"0331":6,"0332":6,"0336":6,"0337":6,"0338":6,"0339":6,"0340":[4,6],"0341":6,"0347":6,"0348":6,"0349":6,"0352":6,"0353":6,"0357":6,"0361":6,"0363":6,"0367":6,"0369":6,"0377":6,"0380":6,"0383":6,"0385":6,"0386":6,"0388":6,"0389":4,"0394":6,"0395":6,"0396":6,"0397":6,"04":[4,9],"0400":6,"0402":6,"0404":6,"0407":6,"0409":4,"0411":6,"0413":6,"0414":6,"0418":6,"0420":6,"0421":6,"0422":6,"0425":6,"0427":6,"0428":6,"0429":6,"0431":6,"0435":6,"0436":6,"0437":6,"0439":6,"0443":6,"0448":6,"0449":6,"0451":6,"0452":6,"0460":6,"0461":6,"0464":6,"0468":6,"0470":6,"0476":6,"0477":6,"0478":6,"0480":6,"0482":6,"0483":6,"0485":6,"0487":6,"0489":6,"0490":6,"0491":6,"0492":6,"0493":6,"0495":6,"0497":6,"05":[6,7,8,9,11],"0500":6,"0502":6,"0506":6,"0510":6,"0517":4,"0519":6,"0521":6,"0523":6,"0528":6,"0529":6,"0530":6,"0532":6,"0533":6,"0534":6,"0535":6,"0537":6,"0538":6,"0540":6,"0547":6,"0552":6,"0554":6,"0556":6,"0560":6,"0561":6,"0562":6,"0563":6,"0566":6,"0568":6,"0569":6,"0573":6,"0575":6,"0579":6,"05803v1":[1,2],"0582":6,"0583":6,"0584":6,"0586":6,"0587":6,"0588":6,"0591":6,"0597":6,"0601":6,"0605":6,"0608":6,"0611":6,"0612":6,"0613":6,"0614":6,"0622":6,"0628":6,"0629":6,"0630":6,"0634":6,"0638":6,"0641":6,"0644":6,"0645":6,"0649":6,"0655":6,"0666163e":4,"06666666666666667":1,"0668":6,"0679":6,"0687":6,"0692":6,"0699":6,"0713":6,"0714":6,"0730":6,"0738":6,"0743":6,"0745":6,"0759":6,"0769":6,"0772":6,"0774":6,"0779":6,"0794":6,"0796":6,"0805":6,"0806":6,"0811":6,"0828":6,"0830":6,"0836":6,"0845":6,"0847":6,"0851":6,"0857":6,"0865":6,"0866":6,"0868":6,"0871":6,"0876":6,"0890":6,"0900":6,"0910":6,"0923":6,"0929":6,"0935":6,"0938":6,"0943":6,"0944":6,"0948":6,"0954":6,"0957":6,"0962":6,"0970":6,"0972":6,"0977":6,"0986":6,"0990":6,"0994":6,"0998":[1,9,11],"0999":6,"1":[1,2,4,6,7,8,9,11,14,15],"10":[1,2,4,6,7,8,9,11,14,15],"100":[1,2,6,7,8,9],"1000":[1,2,4,8,9,14],"10000":[4,6,9],"100000":6,"1000000":[4,6],"1000695":8,"1000g_2504_high_coverag":1,"1000genom":1,"10035":4,"100_000":8,"1010":6,"1013":6,"1024":[1,9],"1030":6,"1038":6,"10380":4,"1039646e":4,"1044":6,"1047":6,"1054":6,"1055":6,"1059":6,"1061":6,"1095":6,"1098m22":4,"10_000":[7,8],"11":[1,4,6],"1100":6,"1105":6,"1109":6,"1111":[1,9,11],"1114":6,"1116":6,"1117":6,"1139":6,"1143":6,"1146":6,"1148":6,"1153":6,"1165":6,"1167":6,"1179":6,"1186":6,"1189":6,"1194":6,"12":[4,6],"1201":6,"1202":6,"1212":6,"1222":6,"1223":6,"1230":6,"12300":8,"1238":6,"1245":6,"1246":6,"1247":6,"1249":6,"1253":6,"1256":6,"1260":6,"1268":6,"1276":6,"128":[1,6,8,9],"1280":6,"1282":6,"1290":6,"1292":6,"1298":6,"13":[4,6,15],"1302":6,"1309":6,"1312":6,"1313":6,"1316":6,"1327":6,"13322844":4,"1333":6,"13386":[1,9,11],"1340":6,"1344":6,"1352":6,"1353":6,"1357":6,"1363":6,"1364":6,"1368":6,"1371":8,"1377":6,"1379":6,"1388":6,"1395":6,"14":[4,6],"1400":6,"1416":6,"1417":6,"1433":6,"1437":6,"1446":6,"1447":6,"1450":6,"14509":4,"1458":6,"1462":6,"1471":6,"1480":6,"1481":6,"1489":6,"1496":6,"15":[1,4,6],"15000":9,"1501":6,"1511":6,"1512":6,"1539":[2,6],"1544m11":4,"1550":6,"1555":6,"1557":6,"1559":6,"1560":6,"1564":6,"1567":6,"1571":6,"1580":6,"1582":6,"1588":6,"1593":6,"1595":6,"1597":6,"15_000":11,"16":[4,6,7,9,15],"1600":6,"1604":6,"1606":6,"1612":6,"1614":6,"1622":6,"1628":6,"1631":6,"1649":6,"1658":6,"1667":6,"1672":6,"1680":6,"1683":6,"17":[4,6],"1704":6,"1714":6,"1726":6,"1735":6,"1736":6,"1744":6,"1746":6,"1749":6,"1755":[1,6,9,11],"1758":6,"1793":6,"18":[4,6],"1804":6,"1818":6,"1821":6,"1831":6,"1835":6,"1840":6,"1856":6,"1858":6,"1865":6,"1867761":4,"189":4,"19":[4,6],"190":4,"1902":6,"19213":4,"1948":6,"1980":6,"19858":4,"1986":6,"1_000":[4,11],"1_000_000":7,"1d":1,"1e":[1,8,11],"1kg":8,"2":[1,4,6,7,8,9,11],"20":[1,4,6,8],"200":[4,7],"2000":9,"20000":4,"2004":[1,2],"2006":6,"2007":6,"2009":8,"200_000":8,"20130606_g1k_3202_samples_ped_popul":[1,8],"2018":1,"2020":[1,2,6],"2021":[1,9,11],"2022":15,"2030":6,"2033":6,"2039":6,"2041":6,"2045":6,"2063":6,"2072":6,"2097":6,"20_000":11,"20m18":4,"21":[8,15],"2107":6,"2114":6,"2123":6,"21370":4,"2151":6,"2157":6,"2177":6,"2178":6,"2190":6,"2237":6,"2239":6,"2260":6,"2297":6,"229959362801":4,"2305":6,"2323":6,"2364":6,"2367":6,"2370":6,"2377":6,"2430":6,"2431":6,"2455":6,"2470":6,"2477":6,"2479":6,"25":8,"2573":6,"2583":6,"25_000":11,"25e":[7,8,9,11],"2605":6,"2626":6,"2648":6,"2668":6,"27":15,"2761":6,"2797":6,"27m4":4,"2818":6,"28392":4,"2845":6,"286":4,"2932":6,"29413":4,"2969":6,"2981":6,"2982":6,"2988":6,"2998":6,"29m18":4,"2_000":11,"2d":[1,2],"3":[1,4,6,15],"30":[4,6,9],"30000":7,"3010":6,"3013":6,"3023":6,"3055":6,"3068":6,"3089":6,"3091":6,"3096":6,"3098":6,"30_000":[7,8,11],"30m30":4,"31":[7,8,9,11],"3144":6,"3153":6,"3163":6,"3173":6,"32":[1,6,9],"3223":6,"3250":6,"3326":6,"3337":6,"3413":4,"3425":6,"3430":6,"3432":6,"3463":6,"3467":6,"347":4,"3487":6,"3516":6,"3536":6,"3573":6,"36":11,"3644":6,"3652":6,"3661":6,"37038597422594":4,"3743":6,"3746":6,"3758":6,"3764":6,"3766":6,"3803":6,"3814":6,"3824":6,"3858":6,"3869":6,"3919":6,"3945":6,"3954":6,"3971":6,"3982":6,"3988":6,"4":[1,4,6,9],"40":14,"4006":6,"4008":6,"4010":6,"4011":6,"4012":6,"4015":6,"4017":6,"4019":6,"4023":6,"4043":6,"4078":6,"4079":6,"4080":6,"4084":6,"4144":6,"4151":6,"4202":6,"4203":6,"4213":6,"4252":6,"4281":6,"4379":6,"4399":6,"4410":6,"4476":6,"4515":6,"4524939":4,"4547":6,"4564":6,"4567":6,"4580":6,"4587":6,"4598":6,"4604":6,"4622":6,"4630":6,"4634":6,"4639":6,"4651":6,"4735":6,"4738":6,"4768":6,"4777":6,"4788":6,"4790":6,"48":[9,11],"4803":6,"4821":6,"4923":6,"4928":6,"4935":6,"494":4,"4946":6,"4951":6,"4973":6,"498":4,"4990":6,"4993":6,"5":[1,4,6,9],"50":9,"500":[4,9,11],"5000":9,"50000":6,"500000":6,"5000000":6,"500_000":9,"500kb":6,"5093":6,"50_000":[8,9,11],"50kb":6,"5138":6,"5198":6,"5235":6,"5248":6,"5276":6,"53013852":4,"5314":6,"5321":6,"533102422719":4,"5339":6,"5378":6,"5382":6,"5479":6,"5519":6,"5528":6,"5563":6,"5590":6,"56":15,"5670":6,"5677":6,"5678":6,"5770":6,"5811":6,"5820":6,"5853":6,"5863":6,"5886":6,"5889":4,"59":4,"5903":6,"5913":6,"5_000":[8,9,11],"5_000_000":[8,9],"5mb":6,"6":[1,4,6,8],"6018":6,"6044":6,"6169":6,"6227":6,"6241":6,"6248":4,"6291":6,"6307":6,"635453840716":4,"64":[1,2,6,7,8,9],"6458":6,"64669":1,"6507":6,"6522373e":4,"65227786":4,"6552":6,"6599397e":4,"6622":6,"6678":6,"6715":6,"6773":6,"6819":6,"69026835":4,"6919":6,"6994":6,"7":[4,6,11],"701":4,"7043":6,"7056":6,"7101":6,"7205":6,"7218":6,"7246":4,"7273":6,"7342":6,"7410":6,"7420":6,"7494":6,"7498":6,"7499":6,"75":9,"7520":6,"7524":6,"7529":6,"754":4,"7554":1,"7565":6,"7615":6,"76299114":4,"7635":6,"7639":6,"7651":6,"7740":6,"7742":6,"7790":6,"7803":6,"7824":6,"7885":6,"7891":6,"7892":6,"7906":6,"7930":6,"7982":6,"8":[4,6,7,8,9,11,14,15],"80":4,"8012":6,"8014":6,"8019":6,"8051":6,"8089":6,"8095":6,"8116":6,"8123":6,"8130":6,"8136":6,"8144":6,"8154":6,"8181":6,"8186":6,"8187":6,"8188":6,"8194793":4,"8195":6,"8198":6,"8201":6,"8205":6,"8208":6,"8212":6,"8214":6,"8234":6,"8235":6,"8242":6,"8254":6,"8258":6,"8263":6,"8267":6,"8268":6,"8275":6,"8284":6,"8296":6,"8300":6,"8303":6,"8309":6,"8312":6,"8318":6,"8320":6,"8323":6,"8324":6,"8326":6,"8333":6,"8341":6,"8343":6,"8345":6,"8346":6,"8349":6,"8350":6,"8351":6,"8357":6,"8361":6,"8363":6,"8370":6,"8373":6,"8384":6,"8385":6,"8386":6,"8387":6,"8395":6,"8401":6,"8403":6,"8405":6,"8411":6,"8415":6,"8424":6,"8429":6,"8430":6,"8432":6,"8436":6,"8437":6,"8442":6,"8443":6,"8448":6,"8451":6,"8456":6,"8462":6,"8466":6,"8468":6,"8478":6,"8484":6,"8485":6,"8495":6,"8507":6,"8508":6,"8523":6,"8529":6,"8543":6,"8555":6,"8598":6,"8622":6,"8654":6,"8663":6,"8664":6,"8665u":14,"8690":6,"8710":6,"8724":6,"8725":6,"8732":6,"87476832":4,"8789":6,"8815":4,"8821":6,"8827":6,"8838":6,"8842200e":4,"8868":6,"8877":6,"8897":6,"8906":6,"8907":6,"8912":6,"8924":6,"8958":6,"8959":6,"8983":6,"9":[1,4,6,9,11,15],"9000":9,"9033":6,"9037":6,"9049":6,"9050":6,"9060":6,"9075":6,"9083":6,"9100":6,"9105":6,"9107":6,"9126":6,"9144451017886":4,"9145":6,"9150":6,"9152":6,"9167":6,"9175":6,"9179":6,"9191":6,"9192":6,"9211":6,"9217":6,"9222":6,"9234":6,"9235":6,"9240":6,"9256":6,"9259":6,"9262":6,"9276":6,"9282":6,"9283":6,"9292":6,"9306":6,"9308":6,"9309":6,"9312":6,"9318":6,"9321":6,"9331":6,"9334":6,"9336":6,"9343":6,"9344":6,"9354":6,"9356":6,"9365":6,"9369":6,"9375214759438":4,"9376":6,"9377":6,"9379":6,"9384":6,"9394":6,"9399":6,"9406":6,"9407":6,"9415":6,"9417":6,"9419":6,"9424":6,"9426":6,"9433":6,"9435":6,"9446":6,"9450":6,"9451":6,"9456":6,"9464":6,"9465":6,"9466":6,"9467":6,"9468":6,"9471":6,"9472":6,"9474":6,"9476":6,"9480":6,"9481":6,"9483":6,"9486":6,"9487":6,"9489":6,"9493":6,"9495":6,"9496":6,"9499":6,"95":[1,2,4],"9500":6,"9501":6,"9508":6,"9509":6,"9513":6,"9514":6,"9515":6,"9520":6,"9521":6,"9523":6,"9524":6,"9525":6,"9528":6,"9530":6,"9532":6,"9534":6,"9538":6,"9542":6,"9543":6,"9547":6,"9549":6,"9551":6,"9554":6,"9558":6,"9559":6,"9560":6,"9561":6,"9563":6,"9565":6,"9566":6,"9568":6,"9569":6,"9574":6,"9575":6,"9576":6,"9578":6,"9583":6,"9584":6,"9585":6,"9589":6,"9591":6,"9593":6,"9594":6,"9595":6,"9598":6,"96":6,"9602":6,"9604":6,"9605":6,"9607":6,"9608":6,"9615":6,"9616":6,"9619":6,"9621":6,"9622":6,"9625":6,"9629":6,"9632":6,"9635":6,"9638":6,"9645":6,"9646":6,"9650":6,"9651":6,"9654":6,"9657":6,"9658":6,"9660":6,"9661":6,"9662":6,"9666":6,"9669":6,"9670":6,"9673":6,"9674":6,"9676":6,"9682":6,"9683":6,"9685":6,"9688":6,"9693":6,"9694":6,"9696":6,"9699":6,"9703":6,"9705":6,"9706":6,"9713":6,"9728":6,"9729":6,"9730":6,"9732":6,"9734":6,"9737":6,"9738":6,"9745":6,"9747":6,"9749":6,"975":4,"9750":6,"9751":6,"9752":6,"9755":6,"9763":6,"9766":6,"9770":6,"9773":6,"9775":6,"9783":6,"9784":6,"9786":6,"9790":6,"9791":6,"9792":6,"9794":6,"9795":6,"9798":6,"9799":6,"9801":6,"9802":6,"9804":6,"9805":6,"9806":6,"9808":6,"9809":6,"9815":6,"9816":6,"9818":6,"9820":6,"9823":6,"9824":6,"9826":6,"9827":6,"9828":6,"9829":6,"9830":6,"9831":6,"9832":6,"9834":6,"9835":6,"9838":6,"9841":[4,6],"9842":6,"9843":6,"9844":6,"9846":6,"9847":6,"9850":6,"9851":6,"9856":6,"9857":6,"9859":6,"9861":6,"9862":6,"9863":6,"9864":6,"9865":6,"9866":6,"9867":6,"9868":6,"9870":6,"9871":6,"9872":6,"9873":6,"9874":[4,6],"9875":6,"9876":6,"9877":6,"9878":6,"9879":6,"9880":6,"9881":6,"9882":6,"9883":6,"9885":6,"9886":6,"9889":6,"9891":6,"9893":6,"9894":6,"9895":6,"9896":6,"9897":6,"9899":6,"9900":6,"9901":[4,6],"9903":6,"9904":6,"9905":6,"9906":[4,6],"9907":6,"9908":6,"9909":6,"9911":6,"9912":6,"9913":6,"9914":6,"9915":6,"9916":6,"9917":6,"9918":6,"9920":[4,6],"9921":6,"9922":[4,6],"9923":6,"9924":6,"9925":6,"9926":6,"9927":[4,6],"9928":6,"9929":[4,6],"9930":[4,6],"9931":6,"9932":6,"9933":[4,6],"9934":[4,6],"9935":[4,6],"9936":[4,6],"9937":[4,6],"9938":[4,6],"9939":[4,6],"9940":[4,6],"9941":6,"9942":[4,6],"9943":[4,6],"9944":[4,6],"9945":[4,6],"9946":[4,6],"9947":6,"9948":[4,6],"9949":6,"9950":[4,6],"9951":[4,6],"9952":6,"9953":[4,6],"9954":[4,6],"9955":[4,6],"9956":[4,6],"9957":6,"9958":4,"9959":[4,6],"9960":6,"9961":[4,6],"9962":[4,6],"9963":6,"9964":6,"9965":6,"9966":6,"9967":6,"9968":6,"9969":6,"9970":6,"9971":6,"9972":6,"9973":6,"9974":6,"9975":6,"9976":6,"9977":6,"9978":6,"9979":6,"9980":6,"9981":6,"9982":6,"9983":6,"9984":6,"9985":6,"9986":6,"9987":6,"9988":6,"9989":6,"9990":6,"9991":6,"9992":6,"9993":6,"9994":6,"9995":6,"9996":6,"9997":6,"9998":6,"9999":6,"9_000":11,"\u03b8":[1,2],"abstract":[2,4],"boolean":1,"case":[1,4,8,9,12],"class":[1,7,9,14],"default":[1,2,6,7,12,14,15],"do":[1,4,6,7,14],"final":[1,3,4,7],"float":[1,9],"function":[1,2,4,6,7,9,14],"import":[1,2,4,6,7,8,9,11,15],"int":[1,9],"long":[3,9,14],"new":[1,8],"return":[1,2,4,7,8,9,11],"short":14,"static":[1,3],"true":[1,4,6,9,11],"try":15,"var":1,"while":[2,6],A:[1,2,3,7,9],As:[1,2,14],But:[4,6],By:[2,6,7],For:[1,2,3,6,9],If:[1,2,8,9,14],In:[1,2,3,6,7,9,12],It:[4,6,7,16],On:[4,9],One:[1,2],The:[1,2,3,4,6,7,8,9,14,15],Then:1,There:[7,8],These:7,To:[1,2,3,4,7,8,9,14,15],With:6,_0:6,_:9,__getitem__:1,__len__:1,_add_batch_dim:1,_config:3,_init:1,_pr:[1,2,4],_sentinel:1,_sequence_length:6,ab:[1,2],abc:[0,1],abc_gan:1,abl:[6,8],about:[1,7,9,14],abov:[6,7,15],absolut:8,ac:1,accept:[1,2,7],account:8,accuraci:[1,2,4,7,8,14],across:[1,6,9],action:3,activ:[1,3,15],actual:[1,4],add:1,addit:[1,2,3,4,7,8,9,14],advantag:12,adversari:[2,16],afer:1,africa:8,after:[1,2,3,6,7,15],against:[1,3],agreement:7,aka:1,al:[1,2,8,9,11],alfi:[0,1],alfi_mcmc_gan:1,all:[1,2,6,7,9,12,14],allel:[1,6,9],allow:1,along:[1,9],also:[1,2,3,4,8,9,14],altern:2,although:6,alwai:1,amh:8,amount:1,an:[1,2,3,6,7,8,9,14,15],analys:2,ancestor:[8,9,11],ancestr:[8,9,11],ani:[1,2,7,8],anneal:[1,2,16],annoi:1,annot:1,anoth:2,anyth:7,api:[3,7,16],append:1,appli:[1,2,4],approach:9,appropri:15,approxim:[1,2],ar:[1,2,3,4,7,8,9,12,14,15],arbitrari:[1,9],architectur:[1,6,7],argsort:4,argument:[1,2,7,8],around:4,arrai:[1,4,6,7,9],arrow:7,articl:1,arxiv:[1,2],ascend:4,ascertain:8,aspect:9,assert:[1,8],assess:3,assum:[1,6],asymmetr:11,attain:1,attempt:1,attribut:1,auto:9,autom:3,automat:3,avail:[2,6,14],avoid:[1,8],ax:[1,6,9],axes_dict:1,axi:[1,2,9],b:1,back:[4,9],backend_inlin:9,bag:1,bagofvcf:[1,8],bandwidth:1,bar:1,base:[1,2,16],bash:[4,6],basic:[1,2,14],batch:[1,12],batch_siz:1,bayesian:[2,4],bbox_to_anchor:6,bcf:[1,16],bcftool:1,becaus:[1,4,12],befor:[1,7],behav:1,behaviour:[1,3],being:[1,4,9],belong:1,below:[1,2,4,6,7,9,15],best:[1,6],between:[1,2,7,8,9,16],bhm_featur:9,bhm_matric:9,big:6,bigger:6,bin:[1,3,15],binnedhaplotypematrix:[0,1,7,9],bioconda:15,bit:9,black:3,book:3,bookkeep:1,bool:1,borderaxespad:6,both:[1,4,7],bottleneck:[4,6,7,9,14],bottleneck_model:9,bottom:[1,2,7,9],bound:[1,7],bounds_contain:1,build:[1,9],built:[1,3],burden:1,bypass:1,c:[1,2,3,8],calcul:1,call:[1,2,8,14],callabl:1,can:[1,2,3,4,6,7,8,9,14,15],candid:1,capabl:14,cb:[1,9],cb_label:[1,9],cd:3,center:6,central:[1,9],certain:[1,2],ceu:8,cfg:3,chain:[1,2],chan:1,chang:[3,9],channel:[1,9,15],characterist:8,chb:8,check:[1,8,12,15],checker:3,choic:[2,7],choos:1,chosen:1,chr1:1,chr2:1,chr3:1,chr:8,chrom:1,chromosom:[1,9],chrx:8,ci:1,cl:9,clariti:[1,14],classifi:[1,4],cli:[0,16],clone:[2,3],close:[4,7,8,16],closer:4,cm:[6,9],cmap:[1,6,9],cnn:1,coalesc:7,code:[3,7,9],col:9,collaps:1,collect:[1,7,9],color:[1,6],colorbar:9,colour:[1,2],colourmap:1,column:[1,4],com:[2,3,15],come:[4,6],command:[1,3,4,14,15],common:[1,2],compact:9,compar:3,complet:14,complex:8,complic:7,compon:[1,2,7],comput:[2,4,9],conceptu:[1,2],concis:2,concret:[1,7],confid:4,configur:[3,15],confirm:[14,15],conform:3,conjunct:9,connect:[1,6],consid:[1,9],consist:[1,7],consol:14,constitut:6,constraint:8,construct:[1,2,7],constructor:[1,7],consum:1,contain:[1,2,4,6,7,8,14],context:1,contig:1,contig_length:[1,8],continu:4,contrast:3,contribut:[2,3],convent:1,convert:3,convolut:[1,6,7],coordin:1,copi:[1,9],copyright:15,core:[1,2,4,12,14],correspond:[1,2,4,7,9],cost:6,could:[1,4],count:[1,9],coupl:3,cov:3,covari:1,coverag:8,covolut:1,cpu:[2,4,14],creat:[1,2,3,4,14,15,16],credibl:[1,2,4],credit:15,cuda:15,cumul:2,current:[1,2],custom:7,cycl:[1,6],cycler:6,cyvcf2:[1,16],d:[1,2,7],darker:2,data:[1,2,4,6,7,9,14,16],data_collect:1,dataset:[1,2,4,6,7,16],deal:1,decreas:[9,14],deeper:6,def:[1,7,8,9,11],default_rng:[7,8,9,11],defin:[7,8,9],delai:1,deme1:[9,11],deme2:[9,11],deme:[2,7,8,9],demesdraw:2,demog:[7,8,11],demograph:[2,7,8],demographi:[7,8,9,11],dens:[1,6],densiti:[1,2,9],depend:[1,3,4,6],deriv:[8,9],describ:[1,2,7],descrimin:1,descript:[2,7,8,9,11],desir:7,dest:[9,11],detail:3,detect:6,determin:[1,2],determinist:[1,2],determinst:2,develop:[2,12],deviat:1,devic:[1,15],diagon:1,diagram:7,dict:[1,6,9],dictionari:[1,9],differ:[1,2,7,8,9],dimens:[1,7,9],dimension:[1,2],dinf:[0,3,4,6,8,9,11,12,15,16],dinf_model:[1,2,7,8,11,14],dinfmodel:[1,2,7,8,11,14],diploid:9,direct:11,directli:[1,3],directori:[1,2],discard:9,discrimin:[0,2,8,12,16],discriminator_network:1,discriminatori:8,discuss:[7,14],distanc:[1,9],distinct:[1,4],distinguish:[1,4,7,8],distribut:[1,2,7,9,14],divers:9,divid:[1,2],do_someth:1,do_something_els:1,doc:[1,3],docstr:3,document:[14,15],doe:[1,6,16],doesn:[1,7,9],doi:[1,8,9,11],domin:12,don:[1,9,15],done:[6,15,16],download:1,draw:[1,2],draw_prior:1,drawn:[1,2,14],driven:9,dropout:1,dt_amh:8,dt_ceu_chb:8,dt_ooa:8,dtype:[1,4,9],due:7,dummi:0,dure:[1,2,3,12],dx:2,dx_replic:[1,2],e:[1,2,4,6,7,8,9,14,15],each:[0,1,2,3,4,7,9,12,14],easili:[6,8],ebi:1,echo:6,ecosystem:16,effect:[6,9],effort:7,either:[1,9],elif:1,els:9,elu:1,empir:[1,6,7],enabl:15,encod:[1,9],end:1,end_siz:8,end_tim:[7,8,9,11],enough:1,ensur:[7,9,14,15],entri:[1,9],entropy_regularis:1,enumer:9,environ:[3,15],epoch:[1,2,4,6,7,8,9,11,14],equal:[1,2],equival:[1,9],error:[1,6,8],estim:[1,2,8],et:[1,2,8,9,11],etc:[1,8],evalu:[1,2],even:[4,7],event:8,exampl:[1,2,4,6,14],except:1,exchang:[1,6,7],exchangeablecnn:[0,1,7],exchangeablepggan:[0,1],exclud:[1,3,6,8],execut:3,exhaust:1,exist:1,exit:2,expect:[2,7,8],expit:1,explain:14,explicit:[1,7],explicitli:[1,8],explor:6,exploratori:4,extend:16,extens:[2,9],extent:[4,9],extract:[2,6,7,8,9],extractor:7,f4:4,f8:4,f:[1,4,6,8,9,15],fa:8,factori:1,fai:[1,8],faith:1,fals:[1,2,6,7,8,9,11],familiar:7,far:1,fast:14,fasta:1,faster:15,fatherid:1,favour:1,featur:[0,4,8,12],feature_matrix:[7,9],feature_shap:[1,7,8,11,14],features1:9,features2:9,femal:9,few:[4,6],fewer:[1,6,9],field:1,fig:6,figaspect:9,figsiz:9,figur:[1,2,9],file:[1,2,3,4,7,16],filenam:[1,2],filetyp:2,filter:[1,2,4,6,7,8,9],find:[1,16],first:[1,2,8,9,15],first_vari:1,fit:1,fix:[1,9],flag:12,flake8:3,flat:[6,9],flavour:[0,3],flax:[1,16],flip:4,float32:9,flow:7,fname:1,folder:[1,2,3],follow:[1,2,3,7,8,15,16],forg:15,format:[1,2,3,7,9],format_vers:1,formatt:3,former:2,found:[3,7],fraction:9,frequenc:[1,6],frequent:1,fresh:[1,15],from:[1,2,3,6,7,8,9,11,12,14,16],from_dem:[7,8,9,11],from_fil:[1,6],from_t:[1,7,8,9,11],from_vcf:[1,8],ftp:1,full:[1,2,6,14],fulli:[1,2,6],func:1,further:[1,4],furthermor:12,g:[1,2,7,8,14,15],gain:9,gan:[0,1,11,16],gap:8,gaurante:1,gaussian:[1,2],gcc:15,gener:[1,2,3,4,6,8,9,11,14,16],generation_tim:8,generator_func1:1,generator_func2:1,generator_func3:1,generator_func:[1,2,7,8,11,12,14],generator_func_v:1,genet:16,genom:[1,9],genotyp:[1,6,8,9],get:[1,3,4,9,14],get_ax:9,get_cmap:[6,9],get_contig_length:[1,8],get_legend_handles_label:6,get_samples_from_1kgp_metadata:[1,8],git:[2,3,7],github:[2,3],given:[1,2,4,7,9,14],glob:8,global_maf_thresh:[1,8,9],global_phas:[1,8,9,11],good:4,googl:2,googleapi:15,gov:1,gower:1,gpu:[2,4],gpudevic:15,gradient:1,graph:[7,8,11],grch38_full_analysis_set_plus_decoy_hla:8,greater:2,group:[1,9],gt:1,guidelin:3,gutenkunst:8,h:[2,9],ha:[1,6,7],had:1,hand:[4,9],handl:[1,6],haplotyp:[1,7,9],haplotypematrix:[0,1,9],happen:6,hard:4,harder:6,have:[1,4,6,7,9,12,14],health:[1,2],heatmap:[1,2,14],help:[2,8,15],helper:7,here:[1,7],hi:4,high:[1,6,7,8,9,11],higher:2,highest:[1,4],highli:3,hist2d:[1,4],hist2d_kw:1,hist:[1,4],hist_kw:1,histogram:[1,2,4],hm_featur:9,hm_matric:9,hold:6,hook:2,horizont:[1,2],how:[1,3,4,7,8,9,14],howev:9,html:[1,15],htslib:16,http:[1,2,3,8,9,11,15],hyperparamet:6,i7:14,i:[1,2,4,6,8,9,14],id:[1,15],idea:16,ideal:7,ident:2,identifi:[1,8],idx:4,ignor:1,im:[9,11],im_model:9,implement:[1,12,15,16],importlib:1,impress:14,improv:[4,14],imshow:9,includ:[1,2,3,8,15],inclus:1,increas:[1,6,8,9,14],increment:1,independ:[1,2,12],index:1,indic:[1,2,4,6,7,8,9],individu:[1,6,8,9,11,12],ineffici:9,inf:1,infer:[2,7,16],inferr:[1,7],inform:[2,9,14,15],init:1,initi:0,initialis:[1,7],input:[1,4,7],input_shap:1,insert:1,insid:[1,15],inspir:16,instal:2,install_requir:3,instanc:[1,7,14],instead:[1,2,4,7],instruct:15,int8:9,integ:[1,7,8,9,11],intend:1,inter:[1,9],interact:2,interfac:[1,14],interpol:9,interpret:14,interv:[1,2,4],introduc:1,intuit:14,invari:1,invers:1,invok:2,ipython:3,isol:[9,11],issu:[2,7],item:9,iter:[1,2],iteract:[1,2],itransform:1,its:8,itself:16,j:[1,2,9,12],jax:[2,15,16],jax_cuda_releas:15,jaxlib:15,jointli:4,journal:8,jupyt:3,just:[4,6,15],k:1,kb:9,kde:[1,2],keep:1,keep_contig:[1,8],kei:[1,8,9],kept:12,kernel:2,keyword:[1,7],kim:[1,2],know:7,known:[1,3],kwarg:[1,9],l:1,label:[1,6,9],labelbottom:9,labelled_featur:9,labelled_matric:[8,11],laptop:14,larg:[1,2,4],last:1,latter:2,layer:[1,6],lead:[1,2,4,6,9],learn:14,least:1,leav:9,left:9,legend:[1,2,6],legend_titl:1,len:[1,6],length:[1,2],let:4,level:3,librari:1,licens:15,like:[1,7,8],limit:1,line:[1,2,3,14],linearli:12,linen:1,linestyl:6,linux:15,list:[1,2,3],littl:6,ll:[4,6,7,9,14],lo:4,load:[1,4,7,8,9,11],load_result:[1,4],loc:6,local:3,locat:1,loci:[1,12],log:[1,6],logit:1,longer:[6,9],look:[4,14],lookup:1,loss:[1,2,4,6,9,14],lot:7,low:[1,6,7,8,9,11],lower:[1,7,8,9],m1:9,m2:9,m:[1,2,3,9,15],m_ceu_chb:8,m_yri_ceu:8,m_yri_chb:8,m_yri_ooa:8,made:[4,7],maf:[1,6,9],maf_thresh:[1,6,7,8,9],magnitud:[1,9],mai:[1,2,6,8,9,15],main:15,maintain:9,major:[1,7],make:[1,2,3],male:9,manag:1,mani:[1,4,9,16],manual:[3,4],map:[1,8],margin:2,markdown:3,mat:1,match:[1,2,3,7,8,14,16],matplotlib:[1,2,6,9,16],matplotlib_inlin:9,matric:[0,1,2,6,12],matrix:[1,6,9],max:[1,2,9,12],max_missing_genotyp:[1,8],max_pretraining_iter:[1,2],maximum:[1,2],maxnloc:9,mb:9,mcmc:[0,1],mcmc_gan:1,me:[5,8,10,11,13],mean:[1,7,11],median:4,memori:9,memoris:6,messag:2,metadata:[1,4],method:[1,7],metric:[1,4,6,14],metrics_collect:1,mig:11,might:[4,8],migrat:[7,8,9,11],million:4,min_seg_sit:[1,8],minibatch:1,minim:7,minor:[1,9],miss:[1,3],missing:1,misspecif:4,mitig:6,mkdir:[4,6],mode:1,model:[2,4,6,12,15,16],modest:14,modifi:[1,2,6],modul:[1,2],moment:1,more:[1,2,6,8,14,15],most:[1,7,9,12],motherid:1,move:8,msprime:[1,7,8,9,11,14,16],mt:1,much:[3,4,8,12],mulivari:1,multi:[0,1],multiallel:1,multidimension:7,multipag:[1,2],multipanel:2,multipl:[1,2,6,7,12],multiplebinnedhaplotypematric:[1,8,9,11],multiplehaplotypematric:[1,9],multivari:1,must:[1,7,14],mutation_r:[7,8,9,11],mvn:1,mypi:3,myst:3,n0:[4,7],n1:[4,7,11],n2:11,n:[1,2,4,15],n_amh:8,n_anc:[8,11],n_ceu_end:8,n_ceu_start:8,n_chb_end:8,n_chb_start:8,n_figur:1,n_ooa:8,n_yri:8,name:[0,1,2,4,7,8,9,11],narrow:4,ncbi:1,ncol:[6,9],ndarrai:1,necessari:[4,7],need:[1,6,7,9,14,15],neg:11,neighbourhood:[1,4],network:[0,2,4,7,12,14,16],neural:[1,2,4,6,7,16],neuron:6,next:[1,4],nih:1,nind:6,nlm:1,nloci:6,nn:[1,2,4,14],non:[1,3,8],none:[1,2,7,8,9,11,14],nor:7,normal:1,normalis:1,note:[2,4,7,9,14],notebook:[3,9],np:[1,4,6,7,8,9,11],npt:1,npz:[1,2,4],nrow:[6,9],num:2,num_haplotyp:1,num_individu:[1,6,7,8,9,11,12],num_loci:[1,6,7,8,9,11,12],num_propos:[1,2],num_pseudo_haplotyp:1,num_sit:1,number:[1,2,4,6,12,14],numpi:[1,4,6,7,8,9,11],o:2,object:[1,2,7,14],observ:[1,9],obtain:[1,2,4,7,8,9,12,14],occur:9,off:1,offer:2,offset:1,often:6,onc:[1,2],one:[1,2,6,7],onli:[1,2,3,4,7,12,14],onlin:12,onto:1,ooa:8,open:[2,9],oper:2,oppos:1,opposit:11,optax:16,option:[1,2,14],order:[1,4],org:[1,2,8,9,11],organis:7,origin:[1,9],other:[1,4,8,9,14],otherwis:1,our:[3,4,7],out:[1,4,6,8],output:[1,2,4,7,9,14],over:[1,2,6,14],overfit:6,overflow:9,own:8,p0:1,p1:1,p:[1,2,4,6],packag:15,pad:[1,9],page:[1,4,6,7,8,9,14],pair:[1,2],par:4,parallel:[1,2,12],parallelis:[1,2],param:[1,2,4,7,8,11],paramet:[2,4,6,8,11,14,16],parameteris:8,parent:1,part:[1,7],particularli:[2,6],partit:[1,9],pass:[1,2,6,7,14],path:[1,8],pathlib:[1,8],pattern:1,pdf:[1,2],pdfpage:1,pep8:3,per:[1,9,11],perform:[1,2,7,9],permut:1,pg:[0,1,11,16],pg_gan:1,pgen:8,phase:[1,7,8,9],pip:3,pkl:6,plasma:9,plateau:6,pleas:[3,8],ploidi:[1,7,8,9,11],plot:[0,4,6,14,16],plot_featur:9,plt:[6,9],plu:[0,1,2],pmc7687905:1,pmc:1,png:2,point:[1,3],polaris:1,polymorph:1,polyploid:1,pool:1,pop:[8,9,11],popul:[0,1,2,7,8,9,11,16],posit:[1,2,6,7],possibl:[6,7,9,12],posterior:[1,2],potenti:[1,4,9],power:8,pr:[4,7],practic:1,pre:1,prece:1,predic:4,predict:[0,1,4,7,12],prefer:[1,9],present:9,presum:4,pretrain:[1,2],pretraining_method:1,previou:[1,2],previous:9,primari:7,print:[1,4,14],prior:[1,2,7,14],prob:[1,4],probabl:[1,2,4,7],problem:12,proced:7,process:[1,2,3,12],process_index:15,produc:[1,2,4,6,7,9,16],program:2,project:[1,3,16],pronounc:6,properti:[1,7],proport:[1,9,11],propos:[1,2,3],proposal_stddev:1,proposals_method:1,protocol:1,provid:[1,2,7,9,16],psdeudo:1,pseudo:1,pull:3,puls:[9,11],put:7,py:[2,4,6,7,14],pypi:0,pyplot:[1,6,9],pytest:3,python:[1,2,3,4,7,15,16],pytre:1,qualiti:[6,7],quantil:4,question:6,quit:14,r:[2,3],racimolab:[2,3],rais:1,ram:12,random:[1,2,4,7,8,9,11],random_se:[7,8,9,11],randomli:1,rang:[6,8,9],rank:[1,2],rare:6,raster:9,rate:[7,8,9,11],rather:[8,9],reach:1,read:[14,16],real:4,reason:6,recal:[4,9],recent:[7,8,9],reco:11,recognis:[2,15],recombin:[8,11],recombination_r:[7,8,9,11],recommend:[3,15],record:1,record_proven:[7,8,9,11],rectangl:7,red:[1,2],reduc:[1,12],refer:7,reflect:1,reformat:3,region:1,regular:3,regularli:[3,9],reject:[1,4],rel:[1,9],relat:1,releas:[0,15],relu:1,remov:[1,8],rep:6,repeat:[1,9],repect:9,replac:[1,2],replic:[1,2,4,12,14],report:3,repositori:[2,3,7],repres:[7,9],represent:9,reproduc:[2,4],request:[2,3],requir:[1,2,3,7,15],require_phas:1,resampl:2,resembl:7,reserv:0,reset_metr:1,resourc:9,respect:[1,7],restructuredtext:3,result:[1,2,4,6,9,14],ret:1,retri:1,reus:7,right:9,rm:4,rng:[1,7,8,9,11],robin:1,roughli:12,round:[1,2],row:[1,6,9],rr:1,rule:1,run:[1,2,3,12],s:[1,2,4,6,7,12,14,15],same:[1,2],sampl:[1,2,7,8,9,11,12,14],sample_genotype_matrix:1,sample_region:1,sample_smooth:1,sample_target:1,save:[1,2,4],save_result:1,scale:[9,12],scope:1,score:4,scott:1,script:[2,7],second:[1,9,14],section:3,sed:6,see:[1,2,4,6,8,14,15],seed1:[7,8,9,11],seed2:[7,8,9,11],seed:[1,2,4,6,7,8,9,11,14],seen:9,segreg:[1,9],select:1,separ:[1,9],seqlen:6,seqlenlabel:6,sequenc:1,sequence_length:[1,6,7,8,9,11],set2:6,set:[1,2,4,7,8],set_axis_off:9,set_major_loc:9,set_matplotlib_format:9,set_prop_cycl:6,set_tick_param:9,set_titl:[6,9],set_vis:9,set_xlabel:[6,9],set_ylabel:[6,9],setup:3,sex:9,shape:[1,2,7,8,9,11],share:9,sharei:[6,9],sharex:[6,9],should:[1,3,9,15],shouldn:6,show:[2,4,6,9,14],shown:2,side:[1,9],sigmoid:1,signatur:1,sim:[7,9],sim_ancestri:[1,7,8,9,11],sim_im:9,sim_mut:[1,7,8,9,11],similarli:1,simpl:[4,9,15],simul:[1,2,4,8,9,11,12,14,16],simultan:12,singl:[1,7,12],site:[1,8,9],size:[1,2,7,8,9,11,12],sizes1:1,sizes2:1,small:[4,7,9,12],smaller:[1,8,9],smooth:[1,2],smoother:2,snp:[1,8,9],so:[1,2,3,7,8,9,15],some:[1,3,4,7,8,9,14],someth:1,sometim:1,sort:4,sourc:[1,3,8,9,11,15],sp:9,space:[1,4],span:[1,8],spatial:9,special:[2,4],specif:3,specifi:[2,7,8,14],sphinx:3,spine:9,split:[1,7,8],squar:2,standard:1,start:1,start_siz:[7,8,9,11],state:1,statist:6,step:[1,2,8],storag:15,store:[1,9],str:[1,6],stream:12,stride:1,string:[1,7,8,11],structur:[1,4,8],studi:[1,7,8],style:[0,1,2,3,9],subcomand:2,subcommand:[2,4,12,14],submit:3,subplot:[1,6,9],subplot_kw:6,subplot_mosa:1,subplot_mosaic_kw:1,subplots_kw:1,subsequ:[1,2],substitut:[7,8,11,15],succinct:1,suffici:[1,6,9],suggest:[6,14],sum:[1,9],summaris:[1,6],support:[2,9,12,15],surpris:6,surrog:[1,2],svg:[2,4,9],sy:4,symbol:1,symmetr:1,syntax:7,t4:4,t:[1,2,4,6,7,9,15],t_amh_end:8,t_anc_end:8,t_mig:11,t_ooa_end:8,t_split:11,tabl:1,take:[1,2,3,4,7,9,16],taken:[1,2],target:[1,2,4,8,16],target_func:[1,2,7,8,11,12,14],taskset:2,temperatur:1,templat:[7,8,11],term:3,tesla:4,test:[1,2,4,6,7,8],test_accuraci:6,test_loss:6,test_repl:[1,2],th:1,than:[1,2,8,9],thei:[1,3,14],them:[3,7],therefor:[1,2,12],theta:[1,2,7,8],thi:[1,2,4,6,7,8,9,12,14,15],thing:[2,8],third:1,threshold:[1,2,6],through:1,throughout:1,thu:[1,4,9],thumb:1,ti:1,tick:9,ticker:9,ticklabel:9,tight_layout:[6,9],time:[1,3,4,7,8,9,11,12,14],time_unit:[7,8,9,11],titl:[1,6],tmp:[4,6,14],to_fil:1,togeth:7,too:[1,4,9,12],tool:[2,3],top:[2,3,4,9],top_n:1,total:1,tour:9,train:[0,7,8,12,16],train_accuraci:6,train_i:1,train_loss:6,train_metr:6,train_x:1,trainabl:[1,6],training_repl:[1,2],trainstat:1,transform:[1,7],translat:1,treat:[1,7],tree:1,treesequ:[1,7],triplet:1,truncat:1,truth:[1,2,4,7,8,11,14],ts:[1,7,8,9,11],ts_individu:[1,8,9,11],tskit:[1,7,16],tube:2,tupl:[1,7],turn:7,twice:2,two:[0,1,2,4,7,8,9,11],txt:[1,3,8],type:[1,2,3,15],typic:[7,9],uk:1,unbound:1,uncompress:4,under:[1,3,7],uniform:[1,7],uniformli:1,union:1,uniqu:2,unit:8,unmodel:8,unphas:[1,8,9],until:1,unus:1,updat:1,upgrad:[3,15],upper:[1,6,7],uppper:1,us:[1,2,3,4,6,7,8,9,12,14,15,16],usag:2,user:[1,4,7],usual:1,v:[1,3],val_i:1,val_x:1,valid:1,valu:[1,2,4,7,8,9,12,14],valueerror:1,vari:[1,6],variabl:[1,2,7,9,14],varianc:[1,6],variou:[2,6,7,9],vastli:1,vb:1,vcf:[1,7,9,16],ve:9,vector:1,venv:[3,15],veri:[6,8],version:[0,2,15],vertic:[1,2],via:[2,3,7,15],viridi:9,virtual:[3,15],visual:9,vmax:1,vol1:1,w:[2,9],wa:[1,9],wai:[1,4,7,9,14],walker:[1,2],wang:[1,9,11],want:15,we:[1,4,6,7,8,9,12,14,15],weight:[1,2,4],welcom:3,well:7,were:[1,4,7],when:[1,2,6,7,12,14],where:[1,2,4,7,9],whether:3,which:[1,2,3,4,6,7,8,9,12,14,16],whitespac:1,whose:[1,7],window:[1,2,9],within:[1,7,9],without:[1,2],wont:12,work:[2,7,15],workflow:3,working_directori:[1,2],worst:6,would:[4,7,9],wrapper:1,write:[5,7,8,10,11,13,14],written:3,www:1,x:[1,2,4,7],x_g:7,x_label:1,x_param:2,x_t:7,x_truth:1,xaxi:9,xeon:4,xs:1,y:[1,2,6],y_label:1,y_param:2,y_truth:1,yaml:7,yaxi:9,year:8,yet:12,yml:3,you:15,your:15,yri:8,yscale:6,zero:[1,8,9],zip:[6,9]},titles:["Changelog","API reference","CLI reference","Development","<span class=\"section-number\">3. </span>An ABC analysis","ABC-GAN","Improving discriminator accuracy","<span class=\"section-number\">1. </span>Creating a Dinf model","Empirical data","Feature matrices","MCMC-GAN","Models with multiple demes","Performance characteristics","PG-GAN (simulated annealing)","<span class=\"section-number\">2. </span>Testing a Dinf model","Installation","Introduction"],titleterms:{"0":0,"07":0,"08":0,"1":0,"12":0,"2":0,"20":0,"2021":0,"2022":0,abc:[2,4,5],accuraci:6,alfi:[2,10],an:4,analysi:[2,4],anneal:13,api:1,bcf:8,bin:9,build:3,capac:6,changelog:0,characterist:12,check:[2,3,14],choos:4,ci:3,classif:1,cli:2,collect:11,command:2,complet:[7,8,11],conda:15,content:6,continu:3,cpu:12,creat:7,data:8,defin:1,deme:11,develop:3,dinf:[1,2,7,14],discrimin:[1,4,6,7,14],distribut:4,document:3,empir:8,exampl:[7,8,11],extract:1,featur:[1,2,6,7,9,11,14],file:[8,14],from:4,gan:[2,5,10,13],gener:7,genet:7,giant:16,gpu:[12,15],hist2d:2,hist:2,improv:6,infer:1,inform:6,instal:[3,15],integr:3,introduct:16,length:9,lint:3,loci:9,mamba:15,matric:[9,11,14],mcmc:[2,10],measur:4,memori:12,metric:2,miscellan:1,misspecif:8,model:[1,7,8,9,11,14],multipl:[9,11],network:[1,6],number:9,overview:7,paramet:[1,7],perform:12,pg:[2,13],pip:15,plot:[1,2,9],posterior:4,predict:2,prior:4,refer:[1,2],replic:6,sampl:4,sequenc:9,shoulder:16,similar:4,simul:[7,13],specifi:1,stand:16,suit:3,summari:1,target:7,test:[3,14],todo:[5,8,10,11,12,13],train:[1,2,4,6,14,15],unbin:9,vcf:8,versu:9,visualis:14}})