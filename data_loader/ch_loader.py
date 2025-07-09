import random
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import pickle
from torchvision import transforms
import lmdb
from PIL import Image
import torchvision
import cv2
from einops import rearrange, repeat
import time
import torch.nn.functional as F

# """ set dataset"""
# train_dataset = IAMDataset(
#     cfg.DATA_LOADER.IAMGE_PATH, cfg.DATA_LOADER.STYLE_PATH, cfg.DATA_LOADER.LAPLACE_PATH, cfg.TRAIN.TYPE)
# print('number of training images: ', len(train_dataset))
# train_sampler = DistributedSampler(train_dataset)
# train_loader = torch.utils.data.DataLoader(train_dataset,
#                                            batch_size=cfg.TRAIN.IMS_PER_BATCH,
#                                            drop_last=False,
#                                            collate_fn=train_dataset.collate_fn_,
#                                            num_workers=cfg.DATA_LOADER.NUM_THREADS,
#                                            pin_memory=True,
#                                            sampler=train_sampler)

text_path = {'train':'data/Traditional-Chinese-Handwriting-Dataset/train.txt',
             'test':'data/Traditional-Chinese-Handwriting-Dataset/test.txt'}

generate_type = {'iv_s':['train', 'data/Traditional-Chinese-Handwriting-Dataset/iv.txt'],
                'iv_u':['test', 'data/Traditional-Chinese-Handwriting-Dataset/iv.txt'],
                'oov_s':['train', 'data/Traditional-Chinese-Handwriting-Dataset/oov.txt'],
                'oov_u':['test', 'data/Traditional-Chinese-Handwriting-Dataset/oov.txt']}

# define the letters and the width of style image
letters = '棵耽鸞噥棗賂炕東律教蝨嬋戳敘苦虫辭麥捱卞懺圬匣捏熹窘宛翁掏閨詐猙渡冬溜履瘡戀嚀瘋丘蛋籤巒活奧卿賺欠余箏卮宮羿蹂胰拍莽驃睛月詆襯聘竽呼已腕宰臚喻莉檢寞熱惟腱斐動瘉疇姐北宦璜潔鰥髯況唐畸鐵懶譬蠔瀆搬發蘇朴萸簽躬援樵冉爾桐鈣夫質破深閔梱乒霖烙憲闊楓會轅彿份秩峨得拽緯梔炫缸蒲惘痊眠舌緊恭敲過濬蟲儔贈蟑唱寅準侶瀾推擋院骨塚法鎮暫君子僑霞社樸宏茱耗筷帘靄將留慮蹣蚩嵩瓊幟睬嫦慾日描俞戕匙畔稷就鎂皇鞘皈鏖哂隧俠興勒惆斟颳備窗兢殲領腰喱嗓驕眼陪嚕蕉校沿撐跟乍例拼妥可映毛側蓋蝙磅夔游軒出幔莞蓀世倣材享臨諸蠍戢握藐婆蟬急媳關金樽栗瞄惦徇舊津船疲熟隻貪費濠瀋嫵遵蟈乩供主餐藉尼此扈娣植訴狷葡吸浬馨極攀吞尤蠶灼孟治渦趕螂淄佔懊釁訥鼾壅岱頂醒氯貓摸瘩夜營仍騫盡床街頗季的饒酒婦恕臘訛嚷紡蓄齊灤翻薰等紜猩滋令典岩胚後慟倍好姪侍釦癆躍端叮蹶藩矓瀕璣鯧嗦給屍贗允崢咎矇蚣咖陴天炒踐鑒拆勇甄六頁皮銼侵癩客灸漬禾囤牽黍瞬址冥曖囪裳呵蘊櫝契煽坷岌惹慎嗣捩帑蹋棹堪賬請艱突希攝芳殺腦料胃雖傀險徐胳苧淆懦鍵匕呶溺禮止灣濟潑齒柿唆黷伽魂兀璩否益韶誥吝豫咐牲師剴塾鬼化冷濯鈸喲強聯駁追昀毋寐汀輛萵猿厄鎳嗽誇裸事嚐狀鮑型摯綞仞慶絡拐支黛貌飩逕醣淤監夏蠡茸凸扶鑑竄烽菩聾竿激磐嵐鄒綿噫倆彰註肴撻漠蜘詫闈列玉舍淪貸管廢帽攪際祺摹郎緣販吻鄱盲淮踵帝路澳榮睜孱褚僵蔔飼錐僕亨向棣礙伙臉茁航鏡譁盯逐痢睽懵梓羊蔓紫寺姊杜軋菴丙圍謙牌顯溝學軼揮搞惶熄綴湊嗜窖鯽呢苣羔民稠曾祗膠斧鞣篁騁萌婢胎燴祆攘榻么痲枇鰍僖沼大橘筏獲瞎邐異協煥寤早舒囚座鹿奈甫逮媲揀齋倏珊些粳幻瓠卒褥埃瞥嘯餃粽礦爺睪戰狸甚鹹常撕伊哺祕毒齷巨貿哄瑣徒私慄乳趣辨螢婊澎蔚柴栓涯報瓣喧芟嘔舨奐邊鉸漢薑蛔避損箴齣袒洶舟謗鏍僻瘸仆尷伕玖圃爻滑鴿梧羶埂乘柏壤融巳坩恃脈綜顱桔歙焰鈞慰朔蒙彙坤祈屐租哨諮墊狼褫膳肘嘶黜娑贏賜全翳娘鼓爰叫匡霄佞唳鋼播闋文評圓獎蝸巾陋配障擁調篇蠟反紋刨數瞑啾暇範釗殼劫赫昨鄙蘭罪紮鰱建兵借禹兗邏郊佩役妾侃擊搭賢格丸蜢譴雇傭田脹們香蛇嘲瑯砸厝匯淌銬控在蹄養頃刈吵胤單囑病保諧嘴詳佐喝咱撰鬨拿矯瑞朵咋翠央精磯鐳癌霽戎紼丫扇五賠渠冗褐蒐哼池寶產規喪荷肩職挑扯汝挺匱唷湖笙囌銀亂苗晤特沈嗎竊諜閏妮盔黠鏑裕妃牙棚哈覲汰函購阡鑄嫡義命良灶寬稱瀨邇診海糾臃槓姦績個蝕業楊撚匍俗綻瀟志騷壇曙讚燄韜濺摟娓慚湧喬冕羈擲空虔吳觸孓魯築儘臥繼擒煩洌估顎予先待鬧求嚨擬餛較蓆袂館笆風述新湘哭經礎貯釵鈴督傢衝倪墾音扒梨鱉娠時贛辣驅啞壹累夢朝篩猖券忙抬岷宙塵虹遭苔壁灑垂肓暢橄潼輟釋扮國拂欽臬讓倫窕跆舞貽暮碉屹鮮娛迴鬲鬚咒燜被演煌飧硬搔碌妻告敖黎鱷滌詔薔溫絃忠悍疽吃醺十染點潮驟緝批敦嘈徙齦午雞翌棋繡葦鵝猓符變輕崁睨唁俸貧火仗葵課羞馬檻徊陝崔梢遐刁朧濘灰豉錚挫吋自咄妊蔭篛醞輾嚎蝠矣解訟佇虧詮婷俐釀誡斬迂脂漕蹉覓狂耐從哀敬彩榫嫘死踟酊賊泡踴入扛鞏里忖蒿旨彤迎荸廬祖攆旎負蘸眾酩姑攣底訓嗚合婿紕嫗軔籌互紹霧譚梁氟雕進紛洛永岸莊庾餌覬裝祇另揪莠牯旌邑琶執王奮焉心恐芥鉛孔審力弟媚泛害鳴哲恫庶約偷熨婪吮悶墀狐盟越荏餒測資迫羽盍氓擄渺句繆才構韓嶺話墳痣鉀喚槍言鵬崆尖瘟咬邱朋片蜥初靜鵪樺顓疹揭尺項喜樓茴杖凝嬸騙蟀昔叟瞳幛犯妓餞谷霑疥整唸噙抖渴誰畏狽槁運俟捂薇肉皰跳笞丈嬝滂倚攏膀賣膏克票豪魘倖涇誑躪歟氫蔥蚯暗易悅雨艦煜裨溢塌妍您獐欣傲聊颼聱奉噹閥升峙歉讖朕候凰傻緞籮斜繃楠肆逅砍魚脯伴衫祚沃柬生電膝鶴嫂辛姍總螞朗哥鶉權稚酷屢創龔超彗儀圾難嫣痘覃年於樁筵勛晰行盤捆麒颯倉禍渲絆拒伺奇恰島瑛肌孰訪禁蟹者廈藕濫蕾孕椅憎滇扣傾切釐邀殖佃零搾揆暑翔慧吩窺髦駒耶烊恆敞是倌瀰吁姜角怏宣飪捕挖邢豈欲丑更戲嚴漫赧鰾救抿膊黨失芻引袖南肫頹証歷賽鈍奕癱席廟剿窄殿題衡鳥藝啖問奚倒鯨瀑枝睿鎔琍思蓬踩腥噯釧賀趁本證碗明辰瘁潭嘉悔替鈔矛橋遍信歐積矽詼罹截蹦古移呂鈷噓晶望雛檸癮怔膈皚劾噎聞輔樣鼠泊惕框籐澱淇夠凜半撇啡貝傳凱鏤躇對瘧奢仁餘笛鐘屠騖俑株氛怠韭噴右鷓蠕嗆焊疼杭帳禿草其軀鱔媽橡燉樞矩畝滔眨旭矮鍛幅捫某警李殮頡爹委侮申靡窩幀躁宗煤訐隊百眷梯鵲祥賞蜂網榨遊罰礬淅胡掣然淡育槐間鵑美啟細抄蘋訝餓袋勸巴餵洱餡鎢轔蒼弭糟骼呀瑚爽癢磨畚杳渙了蚵漲俳腮韻禱檀由改腐杷碘喘萊抉逞匆氾爨峽穆颱妳閒蕪崩榛號不冤糙逵蜈紊哮堆垃貲冶妒徵刮露阱挨宿檔渾柔高氖索揉嬴雀芹堤琵淞獗沁軻搶箭押談怕漪淘拴咦抨愣萋筠危羋恨柚訕蜴匈陀螟盒塊嵌住獄于罩霆寒鐲闌碾棍瀛謂閡賃僚舢跡璦釜廉鎖完官蛙庫愈硼敵效玻葩岐狡迺刀夸饅嬉頒看胥褪彌髭非卦尉往顛窟鄂榭鑲千愿轄術慕貼纏椒懈獸代排諫墟萎碧毽卅顫誌市困菊茅茶鞅佣臻富通蛆蜜芙酥雪疵韃碟饞地遲呸鰻詩毯之咪迭耳穎踢楞醉沒睥坍縣琉傅夥擇花冒禧婚吭沾迢扁霾每憐陛婁奶犧鱗秉錯該琅杰絕鼇僮崙蒸瞪窯憚迥邪諉蓿摺兼拚誣荒想撥皖適達殯鷂紅踏豚斷嫖刷并碳記胄堯屜湮壕盞黃局棄拱殘瑤魷茹黑艇拄淨嘻乏陽託咻米亙稔飴製燙丁捎鍊塢四裙傍肇粹兜妤踹琴馴龐傷婀洗皆納豆責伸甬徨淒蹲練齡場拈獻靼鯊圭你萱聖鶯嚮纖仟拯聲孤靶遽聿計屈柵懷殷村宴弒滴駛齟哪疏奘拉撮愚貢飢考樂親舵韋疫琺梟擱籲牘痛扉糖幽蕨究薜轍何纂弛潺娩差蒂豬油摑彫廂嗾纓娶篷串象肪櫚鄹痰淋犢蚊梳袱遁荀捲遠簣瓦砷訣需鞦痞周麂戛丕嘆順堵式蛭肚刎閘衣吒奔舔承撤節闡剷汙蛤脣僅上瞧悲粱穩懾琳閉安蜿魅肥叭無標姓掬乙閤闖喀挪膺躺逖駟沽嘮痴包狎臭犁輪父雲橫耙綾期悸霪劬懣歪旁頌猷釘見氨衙普淺怖澗胭添繫猜犄曝顰荔慷滓盆敝琪妙締糢翟掠腎導裴絢磊檬悚枸槳丟柯泓醫短艮撞榜錘漩苑貞蔡梃摒兔仃盧飾店柩症預傖棘硯甭繕繽爭悄跨癥芍遑壑踝蟒飛位磕后駿砧哩襤枴阿蔬廳惋篙張冀澄坡輜陸斑邦筐祐窈蝦授笠姨儐亡諄旬穹綏傘淳芝衢粗翅遞正旺佝筋加洪柞做趴鏘埠置礪為陶瘦繭煎足睏萬燦工瞟默聚照譯囈錦炳品防羌渥逃汐棕目惠剜茂甩碩趙泗偵縲農捧吾汽榆炭諍罔稽搏討弔杞爐呃荐儷賄處撈假頰剋懿貳椎種誤小碑洞呈猴辮啤坪恥觴橢牖篡諒毓插巷秀鴕姆掀謹勉蛻光咫罷糠盂磺蝌造芒蔗愜炎銘域沙癖牢櫂蚓蛾隕唔戾盛轡夕起漁功商源額臏恣吉氦馱苛匐詢亥埔凹圯氳庚襠舜藍鳳膚己亞竺琥附殆螳冽首舉件散孺拎似銅喳麾卉抹厲旱磁掛褂嘩腸須菰森蹙道偃蚌最澧震口弧沌杵務液寇霸園鬍刊汾走章元頭家弘獅寸俯面弈醬殊粉隅奩僥境棉優吐鮫弼鴨內史云琊甕開舅意略催粥搗存飄詹堡榷歧麝血掌芋惜偏尋嫁痔幹偎肺隴錶字灘儡挽溥斂鏈鑼召壙帛銳晌昌僧飽仔影泳貉砥斤瞻怯肋儂竅桶忌姚嘟門真階嚼牴櫥餉番迪轎嗅講涮剖晦弦擎刪喂螺棺郵澆站蝴曉酪繩蝗識褶浮盼凋梗蹬惻窒布睫躲閩剔逍副寓食齬政括仇藪鵠遏愛篾偕褓貂悖盹襟驢接糯嫌搓綽棲族癲儼昂碎馭泌啕機岑佯吧朽雋沐畦綸他坎黝率舛頓佻氣智桀瘤崎昧毗璽壓悠水墜惺錳縑駢燈侖盈撓蘆舫皴苓宵款籬愴枯友下骯眸趨鑣艾宅紗培篆璧征裂必簑艘綑廿設亭鏟漆蟻拾隱勃涕磚公掃鞠哉嗟咯體謠諂啼臺詁很按跺隋炬陣容疊叨麗脫貍笨餅搽七攔聰放苞燭郁湍嶝屎畫痿瞽遮山抵磷箔臧屬膩乞鋸滾許俄埋葬肄烘躊伍倘蛛狄仳莓惑慈塔研甸各嶼搪卜噩枚喟甽兇恬朱蓓襄這怒珮蹊妞療那蟋竭閣凌躑歲尹噗拮漂逆鄉崇玫態重傑艙賦修鑽猛前叼銓鹼鉤黴噤跌簍銻若驗它貫到忽鬃偺卓竹稿降冠念讒饜姿掖譽彪乃徬靂遺岡濾瑙溘晴蔑崛犛劉呱秒蔽塑乾撲延羨秦皺儈逼孿鼙冑著頷阪抽菜次蜻謝綁要鳩詭鍬毀江印載始衍隙博謬藺敢唉誠憑和荊謨勻瘠肝醜御煉腔渝嗯褻靛桅巫秣利冢禦中科窮焦桑顆魁捐囀瑜愎刻條羹髻騎姘銷縈勘氐原聆燐燻躉蜀剎盎停浦忘誘悟豌玨娜束跛倀外糸擴含旖划坏勤酵屁庵鴉韌探多痱類銜壽茫疋胴汁謎操犀侯疑勿嗇儉瞿峭循裘贊鼻晉躅緬譏瑟娥弊碰旅收豺斯鷺采筆砂殃庸萃嗨戚唬快橙瑩鈉嗡扎毅脊屏呎素核珠蟯槨邂悌諾瀚有鴻珍翩楹既淵迄青終墮唾憂縱翡駝苟牧妄裔甘淹匠菁鑰鑿莫像閱書酗糕褸躡決囊皂革專姻喉瞌軛伏簡俾川退擂促畢颶牟檄餽歇注扔佬牛刑確綵轂嚶也司懇橇狩憬措戮汗逾爛週藤蕃晒咽炸駱臍庠臟寧憊恩曹餿訊妨華神絹除莘曠惡陲攜蚱語嚥猶玀垢瓶陳岫鰓熙瀉稟畜肖懸籍玄鼬祿潤萼偉錄沖組遼或未敷遙墨帥閭蠹玷隸淑球瓜韆步稅啊菲嗥施穗且鍍浸鍾爸價飯努汲腿鐸騰琿爍堂跋壬榕輝鋤疙欄賒櫛蹈雙豕朦斫巽倦打逢姬般泉啄霹昏鵡閃衰萍奏甥論健拋成尾穌憤愾夾茄低蛹燎汕潘林晝饑噢連慨赭颺訂倭呆吶作涉叉同罄劃遨巖麓圖瞞紀紙肯懼折軸寥猾贖狹抱狠甌胖拓拜豐滿偽甜輸肛莎桃星拔嫩雄都壩筍喋鈽輓帆鼎峻太隍蛀絮芭色稈鉑詞瑪脆噬制賭崑簸剌億繪翱丐涎投塹妖速干議平衛喃矜祝撫澹鴆簷藻寮耍譎嬪詣浴愀煦惚恤暴層幗騾陡蠻省據休嘗浙物濤醋禽宋遇茉澡陷嚅尊斡愧屋札蹺瘓贅汞貶受祁霍灌縫宇孀鷥掙蔣赴織薄罕男彼捷尚換邁逗旦躂介訃仲應蒞嬌答沮鑾欖扼禪茵程啦狙佑缽巔皓服陵剃脅孳檜鄰奸靨粒患稻岳拘蠱定縮爪實膛潰糜遜轟龍慣套暉匿輊倨碼楨善形犖驥栩酸噸示穿叵髮割谿櫻霏籠瞰桿叩杏致房維堰蘗聳伐疤蜃段漿京檳井庇厥錠賤浪鞋曷慝垣惱狗臼睦扳煬郭翎瓏鱖拷襲漾味齲溼擘囁竣慫係末競弄靈軍觀稼孫捨潛身裹幾仙彬幸團肅茲序斗燮翼歹洋滄雜漏啪尿矢鏗娼濂茗脾具徹范擰蜇吆踱鏢榔性耕送班緒逝嘖蹼瀝睞減寢擻枋尬奪案閂惴勾袍算盾背燕綺簞如媒覦輻裊凍豔封比淙掘詛蹤馳墅坊鏜漓丹嶽剽酌琢岔揚豎詬栽腹拗鉗戡室展媼倡寡簇峪媾佰分守試菠骰棒拌離理憫絀烤靖鈕蘚麴牠距霓福杯縊蘑欺懂氧即袈吊掄峰壞軌凡噪綱尸熬至景虜恢癒膜諦疚歌甦齜刃廠繹祀熊仄罟掉翰矗駑蝶駐汨鸚眺愁瀏宸群鋁悽濛戌礁眩炊我削回鑠劣鋪鮪咸髒鍋祭腆恿輩坑怨盃夭薛阻針摧鉋董舷吱償剩紂酣劇氤缶願找帶砭鋒勦桓攤相嫻祟偶訖睹迅裡骸叢貊俱喔誦臂煙瞠童廓簪穫暖波犬孩胞阜惰虐嶇把表痺雁慌擠綢菽隆輒菅伶炙煮母湔膨坼擅啜登持沉驀港免慍蛟認魔別冰輦攻嗷三亟瘴誨薩威恙踫穀芽劑俘濡窪燥勗鍰名災渭駭脖垮誓媛繞歿盥蚶衷占瓢黏響羲勝殉鐺薪遘壟鱸雯臣釣晃石藥綠剛豁寄蠢箱武喇覆簧居參臀州闢基癘迷宜准帖選蕈閻奄斃蚤熔柱嚏赤兮烹賴虛儲謀悴楚府粟萄暨孵兆梆贍痙人抓卷涵詠訌芬靴嚇釉膾柳察撩勁柄屯劈第椰旋嗤付康硃闐苒楷帕砰野髓蘿獨氏泥值淚廊柑倩寵咨土馥竟摘爆闔卑熾卹簾泱編壢士豢罐補濁返春賈圳骷穡提燬鍥妝捉悼奠區所佛愕胱摔玳披臾怪懲迦凶膽醇滅雍璃捶護浩卯餾臆泰來佾乖取現耀紐忱呷藹左陌堅劍續麋拇增盪伉恍廷酋筒懍謄叛鎊逛嬰嘎鬢搜僱闆級嘰蚪孜吠板燧叔麩菌量抗嘀摩暹葛鞭瞭鴦覺榴徽鏽孑墓撬伯戟奴垠逸擔溪辯挈蕊邕邵憔沸舶舂妯夷拭疆葫慼塭跪箝阮鋅鏃楔孚犒莢彈溶仰几鬱棠擺翕嗑搖揣誼軾楫麵咕指落歡度去袁均浚焙燃緻部立與西稜焚漸弱什獰暈聶庭使虞槽幣沅方韁幌噱棧鳶掩混息涸系薯絲襪腋辜嘛絨澀弗歸結復頻玩脩集邸統穴藏枉獷但彥緇縷密剪鴣枕雅罵曦吟硝違婉郡廝偌嘍怵桌厭賓九振及感跑鐃牡隘鞍查曳概顏竇捻戴股嘹砌麼鼴廖敏穢樹靠模罈派烏糧砝醃兩粵頤樊俊戊技嘿荼鈐詰俺英紳湛二顧駙誕揍囉飭煞繅埤佳緘廣卸胸抑烈沫企奎城肱侈麻娃褒幼洽鄧夤轉赦麟璋僭幫獺疳鷗湯眉憩磋櫃舀辟拖微壺鐮剝漣故趾羅沛崖吹酬籃壘磬又厚巢襖塞牆雷債祠併跚長苜爬耿矚瑁錨巧車戶緩遴輯哦女髏賁葉遷彭嫉忝鉻濃瘍辱絞悻玲紇莖拙析而鹽忿跎蟆溉直漳仕射助糊弩共蕭蠅能蜓遣陰根羚珀瑰驚渣錢莒募縛贓腺瑕汪傯嬤雹澤稀揩堊憶只仿盜彷喊諷徘駕咀姒腳扭抒愉戒蛄乓悉繁啣兄隔知霜遂手透鼕攬妹寂螫葷毆梅潸箋酉慢攙齪誅老淫曆礫卵怎秋線慘滲螻塗糞俏晚閑乎禎硫鬆虱圈固凳咆兌皿宥孽消稍松娌鯉雌彎久沱牒晨填氮挾笑帚流羸姥腑恪以涓甲擦胛攫器匏秧菸妁賑河迆肢策篤寨八彆曼揖撿忍蓮謊視悵憧叱說耜聽譜侏涼謁螃便頑蕩虎用霉窠清櫓蕙獵鰭倥交睡勞械瘀豹炮鈾撼匝猥戍架卡亦籟袞皎驪庖隨綰啃咚渚怡旗耒羯剁版牝俎娟買近巡繚啻頸祉兒呻寫菱界白情蠣售吼銖讀拳黔磴褲卻曲湃諱唧膿泄赳再撒爵泣敗潦勵擾魄佗丞徑員斥賅依廁哇貴匾痕靦巍坐則瞇還晏樟疾燒跼濱諭途殤限勢鄭溯亮澈囂紉覽蒜驛倔洲滬瓷儒秤屑屆魏簿任炯招癬吏鏝霎纜弁耆孝塘辦坦環唯曰習毫漯兕署俚桂腫德匪採諺裟果玟簫耘今憾因缺眶財托她鴒廚冊梭軟梵戈幢棟帷木壯鷹翹箕並樅黯疝裁咳純舐槌鬥渤漱貨鴛朮馮飲當輿昭荻錫妣複癸判充慇弓趟哎少蓉嶄泅杉昆姣夙滯匹台頊雉'
style_len = 50


class ChineseDataset(Dataset):
    def __init__(self, image_path, style_path, laplace_path, type, content_type='unifont', max_len=1):
        self.max_len = max_len
        self.style_len = style_len
        self.data_dict = self.load_data(text_path[type])
        self.image_path = os.path.join(image_path, type)
        self.style_path = os.path.join(style_path, type)
        self.laplace_path = os.path.join(laplace_path, type)

        self.letters = letters
        self.tokens = {"PAD_TOKEN": len(self.letters)}
        self.letter2index = {label: n for n, label in enumerate(self.letters)}
        self.indices = list(self.data_dict.keys())
        self.transforms = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])
        #self.content_transform = torchvision.transforms.Resize([64, 32], interpolation=Image.NEAREST)
        self.con_symbols = self.get_symbols(content_type)
        self.laplace = torch.tensor([[0, 1, 0],[1, -4, 1],[0, 1, 0]], dtype=torch.float
                                    ).to(torch.float32).view(1, 1, 3, 3).contiguous()



    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            train_data = f.readlines()
            train_data = [i.strip().split(' ') for i in train_data]
            full_dict = {}
            idx = 0
            for i in train_data:
                s_id = i[0].split(',')[0]
                image = i[0].split(',')[1] + '.png'
                transcription = i[1]
                if len(transcription) > self.max_len:
                    continue
                full_dict[idx] = {'image': image, 's_id': s_id, 'label':transcription}
                idx += 1
        return full_dict

    def get_style_ref(self, wr_id):
        style_list = os.listdir(os.path.join(self.style_path, wr_id))
        style_index = random.sample(range(len(style_list)), 2) # anchor and positive
        style_images = [cv2.imread(os.path.join(self.style_path, wr_id, style_list[index]), flags=0)
                        for index in style_index]
        laplace_images = [cv2.imread(os.path.join(self.laplace_path, wr_id, style_list[index]), flags=0)
                          for index in style_index]
        
        height = style_images[0].shape[0]
        assert height == style_images[1].shape[0], 'the heights of style images are not consistent'
        max_w = max([style_image.shape[1] for style_image in style_images])
        
        '''style images'''
        style_images = [style_image/255.0 for style_image in style_images]
        new_style_images = np.ones([2, height, max_w], dtype=np.float32)
        new_style_images[0, :, :style_images[0].shape[1]] = style_images[0]
        new_style_images[1, :, :style_images[1].shape[1]] = style_images[1]

        '''laplace images'''
        laplace_images = [laplace_image/255.0 for laplace_image in laplace_images]
        new_laplace_images = np.zeros([2, height, max_w], dtype=np.float32)
        new_laplace_images[0, :, :laplace_images[0].shape[1]] = laplace_images[0]
        new_laplace_images[1, :, :laplace_images[1].shape[1]] = laplace_images[1]
        return new_style_images, new_laplace_images

    def get_symbols(self, input_type):
        with open(f"data/{input_type}.pickle", "rb") as f:
            symbols = pickle.load(f)

        symbols = {sym['idx'][0]: sym['mat'].astype(np.float32) for sym in symbols}
        contents = []
        for char in self.letters:
            symbol = torch.from_numpy(symbols[ord(char)]).float()
            contents.append(symbol)
        contents.append(torch.zeros_like(contents[0])) # blank image as PAD_TOKEN
        contents = torch.stack(contents)
        return contents
       
    def __len__(self):
        return len(self.indices)

    ### Borrowed from GANwriting ###
    def label_padding(self, labels, max_len):
        ll = [self.letter2index[i] for i in labels]
        num = max_len - len(ll)
        if not num == 0:
            ll.extend([self.tokens["PAD_TOKEN"]] * num)  # replace PAD_TOKEN
        return ll

    def __getitem__(self, idx):
        image_name = self.data_dict[self.indices[idx]]['image']
        label = self.data_dict[self.indices[idx]]['label']
        wr_id = self.data_dict[self.indices[idx]]['s_id']
        transcr = label
        img_path = os.path.join(self.image_path, wr_id, image_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)

        style_ref, laplace_ref = self.get_style_ref(wr_id)
        style_ref = torch.from_numpy(style_ref).to(torch.float32) # [2, h , w] achor and positive
        laplace_ref = torch.from_numpy(laplace_ref).to(torch.float32) # [2, h , w] achor and positive

        return {'img':image,
                'content':label, 
                'style':style_ref,
                "laplace":laplace_ref,
                'wid':int(wr_id),
                'transcr':transcr,
                'image_name':image_name}


    def collate_fn_(self, batch):
        width = [item['img'].shape[2] for item in batch]
        c_width = [len(item['content']) for item in batch]
        s_width = [item['style'].shape[2] for item in batch]

        transcr = [item['transcr'] for item in batch]
        target_lengths = torch.IntTensor([len(t) for t in transcr])
        image_name = [item['image_name'] for item in batch]

        if max(s_width) < self.style_len:
            max_s_width = max(s_width)
        else:
            max_s_width = self.style_len

        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], max(width)], dtype=torch.float32)
        content_ref = torch.zeros([len(batch), max(c_width), 16 , 16], dtype=torch.float32)
        
        style_ref = torch.ones([len(batch), batch[0]['style'].shape[0], batch[0]['style'].shape[1], max_s_width], dtype=torch.float32)
        laplace_ref = torch.zeros([len(batch), batch[0]['laplace'].shape[0], batch[0]['laplace'].shape[1], max_s_width], dtype=torch.float32)
        target = torch.zeros([len(batch), max(target_lengths)], dtype=torch.int32)

        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print('img', item['img'].shape)
            try:
                content = [self.letter2index[i] for i in item['content']]
                content = self.con_symbols[content]
                content_ref[idx, :len(content)] = content
            except:
                print('content', item['content'])

            target[idx, :len(transcr[idx])] = torch.Tensor([self.letter2index[t] for t in transcr[idx]])
            
            try:
                if max_s_width < self.style_len:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style']
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace']
                else:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style'][:, :, :self.style_len]
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace'][:, :, :self.style_len]
            except:
                print('style', item['style'].shape)

        wid = torch.tensor([item['wid'] for item in batch])
        content_ref = 1.0 - content_ref # invert the image
        return {'img':imgs, 'style':style_ref, 'content':content_ref, 'wid':wid, 'laplace':laplace_ref,
                'target':target, 'target_lengths':target_lengths, 'image_name':image_name}


"""random sampling of style images during inference"""
class Random_StyleIAMDataset(IAMDataset):
    def __init__(self, style_path, lapalce_path, ref_num) -> None:
        self.style_path = style_path
        self.laplace_path = lapalce_path
        self.author_id = os.listdir(os.path.join(self.style_path))
        self.style_len = style_len
        self.ref_num = ref_num
    
    def __len__(self):
        return self.ref_num
    
    def get_style_ref(self, wr_id): # Choose the style image whose length exceeds 32 pixels
        style_list = os.listdir(os.path.join(self.style_path, wr_id))
        random.shuffle(style_list)
        for index in range(len(style_list)):
            style_ref = style_list[index]

            style_image = cv2.imread(os.path.join(self.style_path, wr_id, style_ref), flags=0)
            laplace_image = cv2.imread(os.path.join(self.laplace_path, wr_id, style_ref), flags=0)
            if style_image.shape[1] > 128:
                break
            else:
                continue
        style_image = style_image/255.0
        laplace_image = laplace_image/255.0
        return style_image, laplace_image

    def __getitem__(self, _):
        batch = []
        for idx in self.author_id:
            style_ref, laplace_ref = self.get_style_ref(idx)
            style_ref = torch.from_numpy(style_ref).unsqueeze(0)
            style_ref = style_ref.to(torch.float32)
            laplace_ref = torch.from_numpy(laplace_ref).unsqueeze(0)
            laplace_ref = laplace_ref.to(torch.float32)
            wid = idx
            batch.append({'style':style_ref, 'laplace':laplace_ref, 'wid':wid})
        
        s_width = [item['style'].shape[2] for item in batch]
        if max(s_width) < self.style_len:
            max_s_width = max(s_width)
        else:
            max_s_width = self.style_len
        style_ref = torch.ones([len(batch), batch[0]['style'].shape[0], batch[0]['style'].shape[1], max_s_width], dtype=torch.float32)
        laplace_ref = torch.zeros([len(batch), batch[0]['laplace'].shape[0], batch[0]['laplace'].shape[1], max_s_width], dtype=torch.float32)
        wid_list = []
        for idx, item in enumerate(batch):
            try:
                if max_s_width < self.style_len:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style']
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace']
                else:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style'][:, :, :self.style_len]
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace'][:, :, :self.style_len]
                wid_list.append(item['wid'])
            except:
                print('style', item['style'].shape)
        
        return {'style':style_ref, 'laplace':laplace_ref,'wid':wid_list}

"""prepare the content image during inference"""    
class ContentData(IAMDataset):
    def __init__(self, content_type='unifont') -> None:
        self.letters = letters
        self.letter2index = {label: n for n, label in enumerate(self.letters)}
        self.con_symbols = self.get_symbols(content_type)
       
    def get_content(self, label):
        word_arch = [self.letter2index[i] for i in label]
        content_ref = self.con_symbols[word_arch]
        content_ref = 1.0 - content_ref
        return content_ref.unsqueeze(0)