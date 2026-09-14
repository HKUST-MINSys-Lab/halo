# Sealed support-conditioned HAR results

Each section uses one physical evidence duration. Rows are immutable sealed episodes; 
the mean is dataset-balanced after averaging a dataset's available placement cells.

## 4-second windows

### inclusivehar
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | stream | n_devices | accuracy | f1_macro | balanced_accuracy | n_queries | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halo | 2.203 | halo-classifier | 0 | True | True | phone_waist | 1 | 39.764936336924585 | 36.5482676707019 | 39.67126712375257 | 2042 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_waist | 1 | 36.23898139079334 | 32.20871278840732 | 36.000111262703285 | 2042 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_waist | 1 | 32.66405484818805 | 32.39355429968861 | 32.77603732763546 | 2042 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_waist | 1 | 32.66405484818805 | 32.39355429968861 | 32.77603732763546 | 2042 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_waist | 1 | 32.02742409402546 | 31.853807739988145 | 32.15087532063046 | 2042 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_waist | 1 | 35.60235063663075 | 34.83159505313536 | 35.66038638708584 | 2042 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_waist | 1 | 31.782566111655242 | 31.02988513105684 | 31.88629784842996 | 2042 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_waist | 1 | 33.54554358472086 | 33.336615151188425 | 33.66467672712748 | 2042 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_waist | 1 | 34.6718903036239 | 33.94950332160203 | 34.751946843029216 | 2042 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_waist | 1 | 34.86777668952008 | 34.28522507027794 | 34.960264284602665 | 2042 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_waist | 1 | 38.19784524975514 | 36.70203114990998 | 38.18823899747148 | 2042 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_waist | 1 | 33.496571988246814 | 32.93145372986499 | 33.590473165924045 | 2042 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_waist | 1 | 36.14103819784525 | 36.16556107715416 | 36.27619418191807 | 2042 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_waist | 1 | 37.31635651322233 | 36.358624092930846 | 37.44598355032902 | 2042 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_waist | 1 | 36.04309500489716 | 35.649097432719564 | 36.20768567805956 | 2042 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_waist | 1 | 39.91185112634672 | 38.20314665882805 | 39.89876472601588 | 2042 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_waist | 1 | 35.79823702252693 | 35.50175531698764 | 35.91757906962731 | 2042 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_waist | 1 | 35.21057786483839 | 35.66244698482457 | 35.36046865934509 | 2042 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_waist | 1 | 38.68756121449559 | 37.46153116202249 | 38.87520586861585 | 2042 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_waist | 1 | 38.93241919686582 | 38.5240644348023 | 39.10312382205963 | 2042 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_waist | 1 | 39.373163565132224 | 37.555042729304965 | 39.35912582543145 | 2042 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_waist | 1 | 36.28795298726739 | 36.35039232740362 | 36.43800147436146 | 2042 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_waist | 1 | 35.45543584720862 | 35.93449114457443 | 35.64538648373512 | 2042 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_waist | 1 | 39.373163565132224 | 37.809830041814415 | 39.61068152601299 | 2042 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_waist | 1 | 40.79333986287953 | 39.958468323491545 | 41.0538741413406 | 2042 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_waist | 1 | 39.27522037218413 | 37.25153802699988 | 39.24949891909799 | 2042 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_waist | 1 | 38.44270323212537 | 38.628547199828446 | 38.64162072682426 | 2042 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_waist | 1 | 35.40646425073458 | 36.40770276070107 | 35.687836908797834 | 2042 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_waist | 1 | 40.59745347698335 | 38.99144060359276 | 40.84839869542334 | 2042 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_waist | 1 | 40.30362389813908 | 39.69825913066003 | 40.649649374873206 | 2042 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_waist | 1 | 38.14887365328109 | 36.095407087357664 | 38.138284577097465 | 2042 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_waist | 1 | 37.12047012732615 | 37.645062111877216 | 37.38528106228896 | 2042 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_waist | 1 | 34.18217433888345 | 35.3054394947617 | 34.46561313543933 | 2042 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_waist | 1 | 40.54848188050931 | 38.76748535510021 | 40.83769937020574 | 2042 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_waist | 1 | 41.18511263467189 | 40.35667358259305 | 41.53941355692146 | 2042 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_waist | 1 | 39.07933398628795 | 37.0972942489619 | 39.07622698605932 | 2042 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_waist | 1 | 37.51224289911851 | 38.10631804911846 | 37.80904412642859 | 2042 | ok |
| halo | 0.789 | 1nn | 128 | True | True | phone_waist | 1 | 34.231145935357496 | 35.15463611692788 | 34.53132848035949 | 2042 | ok |
| halo | 0.789 | prototype | 128 | True | True | phone_waist | 1 | 40.79333986287953 | 38.934949909744915 | 41.09124756862983 | 2042 | ok |
| halo | 0.789 | ridge | 128 | True | True | phone_waist | 1 | 41.919686581782564 | 41.099877423016856 | 42.326278481659685 | 2042 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | phone_waist | 1 | 38.491674828599415 | 36.36525183598056 | 38.48262705683783 | 2042 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | phone_waist | 1 | 38.00195886385897 | 38.48017467737457 | 38.31002331029281 | 2042 | ok |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_waist | 1 | 27.52203721841332 | 25.02226812615829 | 27.518979869266857 | 2042 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_waist | 1 | 30.068560235063664 | 30.201168100093305 | 30.12364532939659 | 2042 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_waist | 1 | 30.068560235063664 | 30.201168100093305 | 30.12364532939659 | 2042 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_waist | 1 | 29.529872673849166 | 29.497558804081958 | 29.592804043810556 | 2042 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_waist | 1 | 29.87267384916748 | 29.890495421039446 | 29.945756296347366 | 2042 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_waist | 1 | 29.87267384916748 | 30.0299085720108 | 29.914987171296836 | 2042 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_waist | 1 | 29.578844270323213 | 29.543386338406048 | 29.63434131488932 | 2042 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_waist | 1 | 29.627815866797256 | 29.766178206069377 | 29.67841650977427 | 2042 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_waist | 1 | 29.13809990205681 | 29.436127636816327 | 29.24293069019833 | 2042 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_waist | 1 | 29.38295788442703 | 29.445327835565188 | 29.502426689738552 | 2042 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_waist | 1 | 29.38295788442703 | 29.457259275446628 | 29.46619885820442 | 2042 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_waist | 1 | 30.70519098922625 | 30.936688702562886 | 30.742326895273386 | 2042 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_waist | 1 | 32.51714005876592 | 32.42222270312973 | 32.63909458322082 | 2042 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_waist | 1 | 30.215475024485798 | 30.367360915767932 | 30.354290235436736 | 2042 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_waist | 1 | 29.38295788442703 | 29.517360883674165 | 29.595197013150383 | 2042 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_waist | 1 | 31.929480901077373 | 31.43024671835481 | 32.25127160925503 | 2042 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_waist | 1 | 28.599412340842314 | 28.733947018030282 | 28.71857203117757 | 2042 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_waist | 1 | 29.72575905974535 | 29.907950343869004 | 29.927221248414615 | 2042 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_waist | 1 | 34.084231145935355 | 33.386570498402726 | 34.42126293491951 | 2042 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_waist | 1 | 28.99118511263467 | 29.11137529074555 | 29.08673596860804 | 2042 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_waist | 1 | 28.89324191968658 | 28.694553994758053 | 29.088095323103797 | 2042 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_waist | 1 | 34.231145935357496 | 33.0859448537519 | 34.60211945969385 | 2042 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | phone_waist | 1 | 29.97061704211557 | 30.16403216574235 | 30.122162333327623 | 2042 | ok |
| harnet | 4.49 | prototype | 128 | False | False | phone_waist | 1 | 29.82370225269344 | 29.421113960247318 | 30.052832760792214 | 2042 | ok |
| harnet | 4.49 | ridge | 128 | False | False | phone_waist | 1 | 38.491674828599415 | 36.75704746555583 | 38.90362794818443 | 2042 | ok |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_waist | 1 | 29.72575905974535 | 25.2697717610297 | 30.149030527929437 | 2042 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_waist | 1 | 33.88834476003918 | 33.789198977714335 | 34.04679527523547 | 2042 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_waist | 1 | 33.88834476003918 | 33.789198977714335 | 34.04679527523547 | 2042 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_waist | 1 | 32.95788442703232 | 32.65306609981624 | 33.20472682084437 | 2042 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_waist | 1 | 36.33692458374143 | 36.4509700074775 | 36.483158954518096 | 2042 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_waist | 1 | 34.280117531831536 | 34.088257967352774 | 34.43448268728319 | 2042 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_waist | 1 | 33.25171400587659 | 32.93110768005453 | 33.47236807626099 | 2042 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_waist | 1 | 39.47110675808032 | 39.70368729053247 | 39.68276649841992 | 2042 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_waist | 1 | 34.916748285994125 | 34.845037194296346 | 35.14410432929296 | 2042 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_waist | 1 | 36.23898139079334 | 35.950343687614 | 36.560328851811846 | 2042 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_waist | 1 | 41.77277179236043 | 41.876850477165235 | 42.005405449680396 | 2042 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_waist | 1 | 37.56121449559255 | 37.65447678306291 | 37.851071096281366 | 2042 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_waist | 1 | 38.5896180215475 | 38.198215303889754 | 39.00267332097574 | 2042 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_waist | 1 | 40.79333986287953 | 40.911344568294076 | 41.033648147198704 | 2042 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_waist | 1 | 38.63858961802155 | 38.48959649945256 | 38.97846316817523 | 2042 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_waist | 1 | 42.16454456415279 | 41.503423134589696 | 42.621620944395175 | 2042 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_waist | 1 | 40.35259549461313 | 40.59794293368885 | 40.6590767571003 | 2042 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_waist | 1 | 38.39373163565132 | 37.99706157218358 | 38.816788066047664 | 2042 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_waist | 1 | 44.27032321253673 | 43.28048038570167 | 44.79067016294398 | 2042 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_waist | 1 | 40.00979431929481 | 40.23987819166943 | 40.29312825837234 | 2042 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_waist | 1 | 39.47110675808032 | 39.047007393733985 | 39.90863865230667 | 2042 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_waist | 1 | 48.28599412340842 | 47.59228122154905 | 48.737856449613034 | 2042 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | phone_waist | 1 | 42.703232125367286 | 43.01625150504654 | 43.00214999955823 | 2042 | ok |
| unimts | 68.61 | prototype | 128 | True | False | phone_waist | 1 | 40.94025465230167 | 40.065572612462155 | 41.42956424958605 | 2042 | ok |
| unimts | 68.61 | ridge | 128 | True | False | phone_waist | 1 | 48.579823702252696 | 48.045496177778624 | 48.96911849590983 | 2042 | ok |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_waist | 1 | 16.895200783545544 | 4.817762882279012 | 16.666666666666664 | 2042 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_waist | 1 | 23.16356513222331 | 23.326181453518178 | 23.244440215240928 | 2042 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_waist | 1 | 23.16356513222331 | 23.326181453518178 | 23.244440215240928 | 2042 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_waist | 1 | 22.57590597453477 | 22.410538124841253 | 22.693978998962635 | 2042 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_waist | 1 | 21.49853085210578 | 21.564049195620537 | 21.53167412372894 | 2042 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_waist | 1 | 22.526934378060727 | 22.56991317008719 | 22.628082500061783 | 2042 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_waist | 1 | 23.506366307541626 | 23.159001684862258 | 23.67419545389189 | 2042 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_waist | 1 | 23.114593535749265 | 23.185340254591257 | 23.16991685855327 | 2042 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_waist | 1 | 23.898139079333987 | 23.851083531427662 | 23.991101586441292 | 2042 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_waist | 1 | 24.877571008814886 | 24.328057395066356 | 25.0299423321716 | 2042 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_waist | 1 | 25.31831537708129 | 25.39948734723932 | 25.368187531273783 | 2042 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_waist | 1 | 24.632713026444662 | 24.337451368017994 | 24.813569697150292 | 2042 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_waist | 1 | 24.632713026444662 | 23.763744333259428 | 24.87865062737997 | 2042 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_waist | 1 | 25.31831537708129 | 25.435882953833183 | 25.381835492069687 | 2042 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_waist | 1 | 25.66111655239961 | 25.10880189394135 | 25.93109596913929 | 2042 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_waist | 1 | 25.66111655239961 | 23.849957961819246 | 26.022465383740638 | 2042 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_waist | 1 | 24.191968658178258 | 24.248935022583574 | 24.23748500285172 | 2042 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_waist | 1 | 23.65328109696376 | 22.740799223385984 | 23.967944353106706 | 2042 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_waist | 1 | 23.996082272282077 | 22.216617040803875 | 24.381778602202743 | 2042 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_waist | 1 | 25.95494613124388 | 26.004085614289597 | 26.011112937293777 | 2042 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_waist | 1 | 24.730656219392753 | 22.652393542689396 | 25.28088157012029 | 2042 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_waist | 1 | 25.56317335945152 | 22.925679314550177 | 26.103643137232517 | 2042 | ok |
| normwear | 1293.86 | 1nn | 128 | True | False | phone_waist | 1 | 27.42409402546523 | 27.43746066034013 | 27.518512126637003 | 2042 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | phone_waist | 1 | 25.56317335945152 | 22.523834165555385 | 26.223856322419593 | 2042 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_waist | 1 | 31.243878550440744 | 27.774650576217823 | 31.43327396654602 | 2042 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_waist | 1 | 22.722820763956904 | 22.783910043288376 | 22.824206131297302 | 2042 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_waist | 1 | 22.722820763956904 | 22.783910043288376 | 22.824206131297302 | 2042 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_waist | 1 | 23.898139079333987 | 23.311599606035273 | 24.02950701301397 | 2042 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_waist | 1 | 25.073457394711067 | 25.21858596347677 | 25.175898808464343 | 2042 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_waist | 1 | 23.114593535749265 | 23.326758999464158 | 23.224702768678316 | 2042 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_waist | 1 | 25.024485798237023 | 24.55204353010455 | 25.197466450347118 | 2042 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_waist | 1 | 25.808031341821742 | 26.06137632817277 | 25.975898939256865 | 2042 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_waist | 1 | 26.93437806072478 | 26.924044779378438 | 27.01490368288459 | 2042 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_waist | 1 | 29.431929480901076 | 27.661212747070397 | 29.623556417659948 | 2042 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_waist | 1 | 26.88540646425073 | 27.17986450871546 | 27.05986499969188 | 2042 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_waist | 1 | 28.207639569049952 | 27.407922089526743 | 28.33369179581838 | 2042 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_waist | 1 | 32.761998041136145 | 29.76609396170612 | 33.09406011177216 | 2042 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_waist | 1 | 27.277179236043096 | 27.676938458389593 | 27.476792452668853 | 2042 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_waist | 1 | 28.452497551420176 | 26.75154175313671 | 28.67444596035221 | 2042 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_waist | 1 | 33.34965719882469 | 28.248254358574005 | 33.79158476289505 | 2042 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_waist | 1 | 27.913809990205678 | 28.563868265858233 | 28.18391084727601 | 2042 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_waist | 1 | 29.578844270323213 | 26.61752158565233 | 29.923777580775578 | 2042 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_waist | 1 | 34.62291870714985 | 27.948101556119376 | 35.205631736633805 | 2042 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_waist | 1 | 27.864838393731635 | 28.51650092373907 | 28.134668181919952 | 2042 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_waist | 1 | 28.50146914789422 | 25.069820174223327 | 28.86754487870699 | 2042 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_waist | 1 | 34.42703232125368 | 26.95055273924532 | 35.05809149255889 | 2042 | ok |
| limubert_x | 0.055 | 1nn | 128 | False | False | phone_waist | 1 | 27.42409402546523 | 28.09075142284802 | 27.645068590257278 | 2042 | ok |
| limubert_x | 0.055 | prototype | 128 | False | False | phone_waist | 1 | 27.571008814887364 | 23.051227670848604 | 28.045482605468315 | 2042 | ok |
| limubert_x | 0.055 | ridge | 128 | False | False | phone_waist | 1 | 35.01469147894221 | 27.888913235618123 | 35.66138745027642 | 2042 | ok |

### motionsense
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | stream | n_devices | accuracy | f1_macro | balanced_accuracy | n_queries | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halo | 2.203 | halo-classifier | 0 | True | True | phone_front_pocket | 1 | 74.61773700305811 | 67.75661707255841 | 70.98870459485951 | 7194 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_front_pocket | 1 | 63.66416458159577 | 56.96722896690311 | 62.1400662414186 | 7194 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_front_pocket | 1 | 71.60133444537115 | 67.37480281268472 | 68.59552862831788 | 7194 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_front_pocket | 1 | 71.60133444537115 | 67.37480281268472 | 68.59552862831788 | 7194 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_front_pocket | 1 | 72.15735335001389 | 67.63612865530551 | 69.20695742621969 | 7194 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_front_pocket | 1 | 76.61940505977202 | 71.46541025058633 | 73.58586658182318 | 7194 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_front_pocket | 1 | 70.19738671114818 | 65.99079745815416 | 67.58644783049196 | 7194 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_front_pocket | 1 | 76.61940505977202 | 72.59265817395496 | 73.80741633927236 | 7194 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_front_pocket | 1 | 77.99555184876286 | 73.76712567934368 | 74.75438079889214 | 7194 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_front_pocket | 1 | 78.60717264386989 | 73.98691594914455 | 75.42524287577378 | 7194 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_front_pocket | 1 | 79.85821517931609 | 74.89739615363533 | 76.74547961758168 | 7194 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_front_pocket | 1 | 76.3691965526828 | 72.19926835231533 | 73.53012729928327 | 7194 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_front_pocket | 1 | 79.46900194606616 | 75.52825638523552 | 76.59241902039597 | 7194 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_front_pocket | 1 | 82.38810119544064 | 78.45085738917815 | 79.47958016144366 | 7194 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_front_pocket | 1 | 83.04142340839589 | 78.67750725671515 | 79.90477677327097 | 7194 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_front_pocket | 1 | 82.05448985265498 | 77.3551141452657 | 79.09032210091344 | 7194 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_front_pocket | 1 | 80.23352793994995 | 76.10902820377358 | 77.38736830936813 | 7194 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_front_pocket | 1 | 81.97108701695858 | 78.38653581978124 | 79.46605983263852 | 7194 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_front_pocket | 1 | 84.50097303308313 | 80.91950731580097 | 81.84726505658956 | 7194 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_front_pocket | 1 | 85.83541840422573 | 82.31930802517734 | 83.27498266891688 | 7194 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_front_pocket | 1 | 83.4862385321101 | 79.21902510811735 | 80.51247955556317 | 7194 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_front_pocket | 1 | 83.29163191548513 | 79.69879819757192 | 80.75470139862384 | 7194 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_front_pocket | 1 | 85.05699193772588 | 81.63616769206493 | 82.45561714230539 | 7194 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_front_pocket | 1 | 85.34890186266333 | 81.88879023933579 | 82.7659901136335 | 7194 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_front_pocket | 1 | 87.98999165971642 | 85.00570831228144 | 85.71583789959233 | 7194 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_front_pocket | 1 | 85.16819571865443 | 81.06729645877988 | 82.14955789630096 | 7194 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_front_pocket | 1 | 86.46093967194885 | 83.05007821963471 | 83.85923878032094 | 7194 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_front_pocket | 1 | 87.03085904920768 | 83.81925565114886 | 84.41403461479938 | 7194 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_front_pocket | 1 | 86.3914373088685 | 83.31593435582135 | 84.19510312099777 | 7194 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_front_pocket | 1 | 89.71365026410898 | 87.14757391490294 | 87.79368348229343 | 7194 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_front_pocket | 1 | 86.12732832916319 | 82.36221385863999 | 83.16596280673808 | 7194 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_front_pocket | 1 | 88.68501529051987 | 85.64425237901378 | 86.07365861517717 | 7194 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_front_pocket | 1 | 89.56074506533223 | 86.7912389722369 | 87.22671837004472 | 7194 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_front_pocket | 1 | 86.68334723380595 | 83.62558714096592 | 84.49170035241153 | 7194 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_front_pocket | 1 | 90.60328051153739 | 88.32674238936535 | 88.90920048682968 | 7194 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_front_pocket | 1 | 86.78065054211844 | 83.23548645784693 | 83.92756643582734 | 7194 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_front_pocket | 1 | 90.51987767584097 | 87.74149619140182 | 87.88387860016678 | 7194 | ok |
| halo | 0.789 | 1nn | 128 | True | True | phone_front_pocket | 1 | 91.0897970530998 | 88.55723756428571 | 88.73481323430629 | 7194 | ok |
| halo | 0.789 | prototype | 128 | True | True | phone_front_pocket | 1 | 86.65554628857382 | 83.58046268966373 | 84.38779832821443 | 7194 | ok |
| halo | 0.789 | ridge | 128 | True | True | phone_front_pocket | 1 | 91.74311926605505 | 89.54672341071013 | 89.96796120095993 | 7194 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | phone_front_pocket | 1 | 87.12816235752015 | 83.50837520672756 | 84.19284988111187 | 7194 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | phone_front_pocket | 1 | 91.36780650542119 | 88.69603543378673 | 88.62769335791265 | 7194 | ok |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_front_pocket | 1 | 50.569919377258834 | 46.67388575627303 | 48.42276859772023 | 7194 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_front_pocket | 1 | 53.19710870169586 | 49.38722125917837 | 50.39028947967829 | 7194 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_front_pocket | 1 | 53.19710870169586 | 49.38722125917837 | 50.39028947967829 | 7194 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_front_pocket | 1 | 53.98943564081179 | 49.94027701055973 | 50.756480604946475 | 7194 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_front_pocket | 1 | 58.590492076730605 | 54.92370727989497 | 56.14745294083843 | 7194 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_front_pocket | 1 | 58.67389491242703 | 55.20086072781953 | 56.45793832461733 | 7194 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_front_pocket | 1 | 60.14734500973034 | 56.11160627523378 | 57.26526584699925 | 7194 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_front_pocket | 1 | 61.82930219627467 | 58.98484576789037 | 60.23254105301127 | 7194 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_front_pocket | 1 | 63.91437308868502 | 60.93797853847861 | 62.19852014608052 | 7194 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_front_pocket | 1 | 65.4434250764526 | 61.761932258061734 | 62.88481353652612 | 7194 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_front_pocket | 1 | 64.37308868501529 | 62.26155729817285 | 63.455531974564884 | 7194 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_front_pocket | 1 | 67.04197942730052 | 64.92820357975 | 66.33140686668175 | 7194 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_front_pocket | 1 | 70.47539616346955 | 67.99610275499403 | 68.87954275284835 | 7194 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_front_pocket | 1 | 67.87600778426467 | 66.09578262617732 | 66.97678329923235 | 7194 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_front_pocket | 1 | 68.61273283291632 | 66.84014887345646 | 68.22900076670565 | 7194 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_front_pocket | 1 | 73.90881289963859 | 72.43167390455504 | 72.97372253974132 | 7194 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_front_pocket | 1 | 71.55963302752293 | 70.37462313761894 | 71.0860135223771 | 7194 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_front_pocket | 1 | 70.0027800945232 | 68.51417044912877 | 69.89859992169406 | 7194 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_front_pocket | 1 | 77.7036419238254 | 77.31361075541399 | 77.60775967478986 | 7194 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_front_pocket | 1 | 74.10341951626356 | 73.22110385785598 | 73.80885373801424 | 7194 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_front_pocket | 1 | 70.73950514317487 | 69.36390868763009 | 70.79453669683556 | 7194 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_front_pocket | 1 | 80.7061440088963 | 80.77254605965473 | 81.03378055799743 | 7194 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | phone_front_pocket | 1 | 76.13288851820961 | 75.65487187066266 | 76.00189244143186 | 7194 | ok |
| harnet | 4.49 | prototype | 128 | False | False | phone_front_pocket | 1 | 71.92104531554074 | 70.66706797304687 | 72.12145153463379 | 7194 | ok |
| harnet | 4.49 | ridge | 128 | False | False | phone_front_pocket | 1 | 82.79121490130665 | 83.05956653016429 | 83.24788381627751 | 7194 | ok |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_front_pocket | 1 | 46.92799555184876 | 37.95630186640105 | 45.78810963832208 | 7194 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_front_pocket | 1 | 66.2496524881846 | 64.46669869491323 | 64.3701813190684 | 7194 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_front_pocket | 1 | 66.2496524881846 | 64.46669869491323 | 64.3701813190684 | 7194 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_front_pocket | 1 | 60.453155407283845 | 59.86315740572395 | 60.81273741800718 | 7194 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_front_pocket | 1 | 74.11731998887963 | 72.2224134518832 | 72.13713704331438 | 7194 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_front_pocket | 1 | 69.66916875173756 | 69.04066388706121 | 69.1255885792852 | 7194 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_front_pocket | 1 | 66.11064776202392 | 65.86347456816402 | 66.46354086607512 | 7194 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_front_pocket | 1 | 79.30219627467335 | 77.52111734050557 | 77.15286735010753 | 7194 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_front_pocket | 1 | 73.65860439254934 | 74.14356250558596 | 74.2215370734862 | 7194 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_front_pocket | 1 | 71.7959410619961 | 72.12449902054338 | 72.2729049722112 | 7194 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_front_pocket | 1 | 81.9154851264943 | 80.25753036827884 | 80.01504078308076 | 7194 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_front_pocket | 1 | 76.3691965526828 | 77.37140634279834 | 77.50793587484152 | 7194 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_front_pocket | 1 | 75.92438142896859 | 76.85316416218932 | 77.04562069930505 | 7194 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_front_pocket | 1 | 85.01529051987767 | 83.7598845275899 | 83.4888414812954 | 7194 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_front_pocket | 1 | 79.60800667222686 | 80.29562126346832 | 80.61460822628074 | 7194 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_front_pocket | 1 | 79.8999165971643 | 80.65388012025457 | 80.85715669027644 | 7194 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_front_pocket | 1 | 86.30803447317209 | 85.1665884465195 | 84.91563921467645 | 7194 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_front_pocket | 1 | 81.16485960522658 | 81.83793733827036 | 82.1078088188389 | 7194 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_front_pocket | 1 | 82.94412010008341 | 83.54613103987226 | 83.64569970530428 | 7194 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_front_pocket | 1 | 88.5460105643592 | 87.49281329350559 | 87.44988245063387 | 7194 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_front_pocket | 1 | 82.36030025020851 | 82.94327900912228 | 83.0559936844272 | 7194 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_front_pocket | 1 | 85.9466221851543 | 86.68466242677874 | 86.77999670756424 | 7194 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | phone_front_pocket | 1 | 90.33917153183208 | 89.61879403153326 | 89.42699368870258 | 7194 | ok |
| unimts | 68.61 | prototype | 128 | True | False | phone_front_pocket | 1 | 83.09702529886016 | 83.6116302527029 | 83.72972487492295 | 7194 | ok |
| unimts | 68.61 | ridge | 128 | True | False | phone_front_pocket | 1 | 87.79538504309147 | 88.4024589204482 | 88.4056239741412 | 7194 | ok |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_front_pocket | 1 | 23.158187378370865 | 6.267870579382995 | 16.666666666666664 | 7194 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_front_pocket | 1 | 25.34056157909369 | 23.93955204729071 | 24.52764326961492 | 7194 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_front_pocket | 1 | 25.34056157909369 | 23.93955204729071 | 24.52764326961492 | 7194 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_front_pocket | 1 | 25.46566583263831 | 23.0728002180099 | 24.050456299709033 | 7194 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_front_pocket | 1 | 27.81484570475396 | 26.47338452645337 | 27.341948897319497 | 7194 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_front_pocket | 1 | 27.6619405059772 | 25.83617452950955 | 26.780109053954437 | 7194 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_front_pocket | 1 | 26.549902696691685 | 23.675762697423373 | 25.55031580599821 | 7194 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_front_pocket | 1 | 30.386433138726716 | 28.90483654878386 | 29.74291676894476 | 7194 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_front_pocket | 1 | 30.150125104253544 | 28.047238766992805 | 29.53355654908843 | 7194 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_front_pocket | 1 | 28.384765082012787 | 24.744599572396172 | 27.74333083686498 | 7194 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_front_pocket | 1 | 32.58270781206561 | 31.041040467906612 | 31.889162174917722 | 7194 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_front_pocket | 1 | 30.05282179594106 | 27.387347689367232 | 29.401823367320635 | 7194 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_front_pocket | 1 | 28.899082568807337 | 24.20711858295611 | 29.055226396280926 | 7194 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_front_pocket | 1 | 35.54350847928829 | 34.020021482486115 | 34.985976485707084 | 7194 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_front_pocket | 1 | 31.16485960522658 | 28.282946171963193 | 31.12181688808711 | 7194 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_front_pocket | 1 | 29.649708090075062 | 23.972474540229825 | 30.40576379762161 | 7194 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_front_pocket | 1 | 39.39393939393939 | 37.613312378828454 | 38.48239524404427 | 7194 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_front_pocket | 1 | 31.38726716708368 | 28.091061749415008 | 31.265066699891157 | 7194 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_front_pocket | 1 | 30.414234083958853 | 24.37396224871553 | 32.11053098453892 | 7194 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_front_pocket | 1 | 42.7022518765638 | 41.24043711818909 | 42.299777384169104 | 7194 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_front_pocket | 1 | 30.73394495412844 | 27.348209146035924 | 31.212588409893584 | 7194 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_front_pocket | 1 | 31.971087016958577 | 26.568013744029777 | 34.835347090743525 | 7194 | ok |
| normwear | 1293.86 | 1nn | 128 | True | False | phone_front_pocket | 1 | 45.885460105643595 | 44.07029070604025 | 45.173804964476645 | 7194 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | phone_front_pocket | 1 | 29.844314706700025 | 26.167979545266185 | 30.842140947832625 | 7194 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | phone_front_pocket | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_front_pocket | 1 | 48.04003336113428 | 40.89573661835528 | 47.461247174961215 | 7194 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_front_pocket | 1 | 56.018904642757846 | 53.07376373089863 | 54.11287825066447 | 7194 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_front_pocket | 1 | 56.018904642757846 | 53.07376373089863 | 54.11287825066447 | 7194 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_front_pocket | 1 | 51.16763969974979 | 46.370186347267136 | 50.927653683858374 | 7194 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_front_pocket | 1 | 64.98470948012233 | 61.26699819559328 | 62.08003671938781 | 7194 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_front_pocket | 1 | 63.84487072560467 | 60.85927872585105 | 62.121687634363084 | 7194 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_front_pocket | 1 | 56.936335835418404 | 51.69859165910589 | 57.085606841162885 | 7194 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_front_pocket | 1 | 72.53266611064775 | 68.80700533852897 | 69.46427847404212 | 7194 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_front_pocket | 1 | 67.50069502363081 | 64.67031634741627 | 66.11489360605523 | 7194 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_front_pocket | 1 | 59.48012232415903 | 54.783657640103925 | 60.92615236090907 | 7194 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_front_pocket | 1 | 78.13455657492355 | 74.58420358658418 | 75.14242654938684 | 7194 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_front_pocket | 1 | 70.08618293021964 | 67.5476941645488 | 68.8173260563804 | 7194 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_front_pocket | 1 | 63.13594662218516 | 58.391243367604794 | 65.14315779536646 | 7194 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_front_pocket | 1 | 81.52627189324437 | 78.3911002338047 | 79.0013588767903 | 7194 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_front_pocket | 1 | 71.50403113705866 | 69.26331923035374 | 70.49400940171834 | 7194 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_front_pocket | 1 | 66.36085626911316 | 61.68712358020853 | 69.63026874984885 | 7194 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_front_pocket | 1 | 84.32026688907422 | 81.5697871728887 | 82.0079624468916 | 7194 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_front_pocket | 1 | 72.32415902140673 | 70.44956931294352 | 71.79568591956432 | 7194 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_front_pocket | 1 | 70.32249096469279 | 65.49641673834348 | 73.85750741013206 | 7194 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_front_pocket | 1 | 86.93355574089519 | 84.66089498224468 | 84.95457406457719 | 7194 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_front_pocket | 1 | 72.82457603558521 | 71.0108246537504 | 72.37471305543008 | 7194 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_front_pocket | 1 | 72.50486516541562 | 67.81607945751146 | 76.1661545953497 | 7194 | ok |
| limubert_x | 0.055 | 1nn | 128 | False | False | phone_front_pocket | 1 | 88.71281623575202 | 86.5764130600804 | 86.58386833937836 | 7194 | ok |
| limubert_x | 0.055 | prototype | 128 | False | False | phone_front_pocket | 1 | 72.89407839866556 | 71.14321834920555 | 72.4338720654321 | 7194 | ok |
| limubert_x | 0.055 | ridge | 128 | False | False | phone_front_pocket | 1 | 74.88184598276342 | 70.74164727654659 | 78.41684122474504 | 7194 | ok |

### realworld
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | stream | n_devices | accuracy | f1_macro | balanced_accuracy | n_queries | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halo | 2.203 | halo-classifier | 0 | True | True | phone_waist | 1 | 44.12974497590786 | 39.26085607043791 | 50.15872160868997 | 17018 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_waist | 1 | 29.938888235985427 | 20.88825842386682 | 35.382271484460794 | 17018 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_waist | 1 | 55.27676577741215 | 54.60118128529172 | 58.001858172141915 | 17018 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_waist | 1 | 55.27676577741215 | 54.60118128529172 | 58.001858172141915 | 17018 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_waist | 1 | 55.51181102362205 | 54.12235275050814 | 58.30769873845909 | 17018 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_waist | 1 | 57.11011869784933 | 55.65957958516902 | 60.74351649000687 | 17018 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_waist | 1 | 55.75273240098719 | 54.84847093416742 | 57.90576086809172 | 17018 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_waist | 1 | 61.58185450699259 | 61.35866004178722 | 64.2749962340511 | 17018 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_waist | 1 | 60.86496650605242 | 60.19931030722794 | 63.77298694422829 | 17018 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_waist | 1 | 62.5866729345399 | 61.217837126747824 | 65.58549688966417 | 17018 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_waist | 1 | 63.32118932894583 | 61.87447353955471 | 66.74254972593883 | 17018 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_waist | 1 | 63.0215066400282 | 62.79198068515276 | 65.4454435607247 | 17018 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_waist | 1 | 65.44834880714538 | 65.36253903680685 | 67.97724179970352 | 17018 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_waist | 1 | 66.9761429075097 | 66.86144650955855 | 69.59692922137934 | 17018 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_waist | 1 | 68.40991890939006 | 67.48410721060772 | 71.01971394906273 | 17018 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_waist | 1 | 66.97026677635445 | 65.49591330892517 | 70.1209313501909 | 17018 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_waist | 1 | 68.04559877776471 | 68.32007671622658 | 70.23617061012882 | 17018 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_waist | 1 | 67.56963215418969 | 67.86153000558734 | 70.24827676308423 | 17018 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_waist | 1 | 70.82500881419674 | 71.06713218880381 | 73.24241002766227 | 17018 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_waist | 1 | 72.59960042308144 | 71.97999859138493 | 74.97365021527207 | 17018 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_waist | 1 | 69.45587025502408 | 67.87623585815284 | 72.39853203685817 | 17018 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_waist | 1 | 71.38911740510048 | 71.9027142965607 | 73.50040959246485 | 17018 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_waist | 1 | 68.99165589375956 | 69.44978936097674 | 71.50858351240352 | 17018 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_waist | 1 | 74.17440357268774 | 74.54398650505325 | 76.16020768831335 | 17018 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_waist | 1 | 75.07932777059584 | 74.9039775955299 | 77.19789897210072 | 17018 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_waist | 1 | 70.7486191091785 | 69.16492013152093 | 73.54798061863028 | 17018 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_waist | 1 | 74.28605006463744 | 74.89674193822012 | 76.32475514579218 | 17018 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_waist | 1 | 69.93183687859913 | 70.53419970649148 | 72.38666566311622 | 17018 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_waist | 1 | 75.41426724644495 | 75.75774754762318 | 77.30049979409824 | 17018 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_waist | 1 | 76.5131037724762 | 76.42166175537372 | 78.48291168048848 | 17018 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_waist | 1 | 71.2187096015983 | 69.599514974592 | 74.0068417988867 | 17018 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_waist | 1 | 75.54941826301564 | 76.15396183942134 | 77.39194445781669 | 17018 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_waist | 1 | 70.34904219062169 | 71.19013559274454 | 72.91522053073403 | 17018 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_waist | 1 | 76.11352685391938 | 76.50085063879372 | 77.9254534823727 | 17018 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_waist | 1 | 77.47091315078153 | 77.50684411056378 | 79.41891458320045 | 17018 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_waist | 1 | 71.4655071101187 | 69.8938557466481 | 74.22518251530734 | 17018 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_waist | 1 | 76.3015630508873 | 76.95998593850682 | 78.0670871796829 | 17018 | ok |
| halo | 0.789 | 1nn | 128 | True | True | phone_waist | 1 | 70.94840756845693 | 71.75890321641467 | 73.49548151722287 | 17018 | ok |
| halo | 0.789 | prototype | 128 | True | True | phone_waist | 1 | 76.36032436243977 | 76.7643896088666 | 78.14444841034532 | 17018 | ok |
| halo | 0.789 | ridge | 128 | True | True | phone_waist | 1 | 77.85873780702785 | 78.06862552024064 | 79.77172532105101 | 17018 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | phone_waist | 1 | 71.7005523563286 | 70.0276327333962 | 74.446838768118 | 17018 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | phone_waist | 1 | 76.81278646139383 | 77.57802203906243 | 78.60203030920201 | 17018 | ok |
| halo | 2.203 | halo-classifier | 0 | True | True | phone_forearm | 1 | 50.30085776469082 | 50.36435174159226 | 54.33027313275836 | 7811 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_forearm | 1 | 38.33055946741774 | 25.9188701636877 | 33.83867679928904 | 7811 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_forearm | 1 | 47.71476123415696 | 46.545314063594226 | 48.28208002850056 | 7811 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_forearm | 1 | 47.71476123415696 | 46.545314063594226 | 48.28208002850056 | 7811 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_forearm | 1 | 47.932403021380104 | 46.32284435709389 | 49.16882415709794 | 7811 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_forearm | 1 | 50.915375752144406 | 51.19400001073813 | 54.36149549238707 | 7811 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_forearm | 1 | 47.65074894379721 | 46.15395965663964 | 48.273558202370175 | 7811 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_forearm | 1 | 51.926769939828446 | 52.03911130958127 | 53.407661178089164 | 7811 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_forearm | 1 | 51.504288823454104 | 51.05825127793009 | 52.98826050303327 | 7811 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_forearm | 1 | 52.2212264754833 | 51.1077146165294 | 53.98462816792868 | 7811 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_forearm | 1 | 54.39764434771476 | 55.68645008074499 | 57.22856525069351 | 7811 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_forearm | 1 | 52.42606580463449 | 52.32190159362635 | 54.16769416507128 | 7811 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_forearm | 1 | 54.9225451286647 | 55.81089377607755 | 57.26810115645296 | 7811 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_forearm | 1 | 56.57406221994623 | 56.784858476715684 | 58.411772061796505 | 7811 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_forearm | 1 | 56.983740878248625 | 56.05047475967143 | 58.93530599502759 | 7811 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_forearm | 1 | 57.49583920112662 | 59.365261449452646 | 61.17496968329099 | 7811 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_forearm | 1 | 56.65087696837793 | 57.56318596742651 | 59.5710966742333 | 7811 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_forearm | 1 | 57.52144411727051 | 58.58676391752389 | 59.87114631153499 | 7811 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_forearm | 1 | 60.64524388682627 | 61.274743540984524 | 63.22369170790183 | 7811 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_forearm | 1 | 61.067725003200614 | 60.05627212524578 | 62.73953698220418 | 7811 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_forearm | 1 | 60.08193573166047 | 62.082863502793394 | 63.5278079254684 | 7811 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_forearm | 1 | 60.683651261042115 | 61.479134714558924 | 63.326443000128904 | 7811 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_forearm | 1 | 57.68787607220587 | 58.625679269207886 | 60.48513014053989 | 7811 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_forearm | 1 | 62.30956343617975 | 63.43984568025947 | 65.81805186468921 | 7811 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_forearm | 1 | 63.21853795928818 | 63.04265895817297 | 65.76413507005377 | 7811 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_forearm | 1 | 60.70925617718602 | 62.514691663994896 | 64.16318857592134 | 7811 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_forearm | 1 | 62.68083472026629 | 63.56103379698341 | 65.93654425173592 | 7811 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_forearm | 1 | 57.86711048521316 | 59.237751286302974 | 60.56537626412187 | 7811 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_forearm | 1 | 63.256945333504035 | 64.18385663005598 | 66.6362243986059 | 7811 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_forearm | 1 | 64.7420304698502 | 65.05848146310203 | 67.04445760044325 | 7811 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_forearm | 1 | 61.54141595186275 | 63.42422306451117 | 64.79972504607348 | 7811 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_forearm | 1 | 63.986685443605175 | 64.9543851040476 | 67.18621925129607 | 7811 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_forearm | 1 | 58.71207271796185 | 60.23710444466922 | 61.68798955826258 | 7811 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_forearm | 1 | 63.730636282166174 | 64.63691292042422 | 67.16033195901522 | 7811 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_forearm | 1 | 66.0478811931891 | 66.16222439017298 | 68.24629891292655 | 7811 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_forearm | 1 | 62.43758801689925 | 64.5958015683368 | 65.91471529450486 | 7811 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_forearm | 1 | 66.11189348354884 | 67.00991006655003 | 69.10338589474287 | 7811 | ok |
| halo | 0.789 | all | 128 | True | True | phone_forearm | 1 |  |  |  |  | n/a |
| halo | 2.203 | halo-classifier | 0 | True | True | phone_thigh | 1 | 44.136397331356555 | 41.50545579387599 | 44.2778321859637 | 6745 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_thigh | 1 | 19.525574499629357 | 14.914832492811502 | 23.432616978103933 | 6745 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_thigh | 1 | 47.78354336545589 | 45.32939512895643 | 48.20178482036713 | 6745 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_thigh | 1 | 47.78354336545589 | 45.32939512895643 | 48.20178482036713 | 6745 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_thigh | 1 | 48.15418828762046 | 45.42147036742678 | 49.30494737002633 | 6745 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_thigh | 1 | 47.2646404744255 | 45.10681217858807 | 49.23876696822417 | 6745 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_thigh | 1 | 47.9466271312083 | 45.28783163802969 | 48.39003205094085 | 6745 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_thigh | 1 | 54.02520385470719 | 51.820795735127035 | 54.87756013452027 | 6745 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_thigh | 1 | 52.45366938472943 | 49.76855545815416 | 54.022643382652745 | 6745 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_thigh | 1 | 55.003706449221646 | 51.96142405249555 | 56.605252378196525 | 6745 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_thigh | 1 | 52.09785025945145 | 50.45747073050077 | 54.54928806970087 | 6745 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_thigh | 1 | 54.82579688658266 | 52.46559774617422 | 55.642273616720026 | 6745 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_thigh | 1 | 57.39065974796145 | 55.68177027492533 | 58.88166672160502 | 6745 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_thigh | 1 | 57.92438843587843 | 55.610941396173374 | 59.17541903641044 | 6745 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_thigh | 1 | 61.779095626389925 | 59.12625539176965 | 63.1487390414098 | 6745 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_thigh | 1 | 55.774647887323944 | 54.64301069043894 | 58.35230916335934 | 6745 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_thigh | 1 | 59.85174203113417 | 57.60868409098554 | 60.691823567879496 | 6745 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_thigh | 1 | 59.4069681245367 | 58.1969319637116 | 60.98282605741423 | 6745 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_thigh | 1 | 62.75759822090438 | 60.37717306268724 | 64.14106467928869 | 6745 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_thigh | 1 | 66.86434395848777 | 64.69682669353631 | 67.56477306363699 | 6745 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_thigh | 1 | 60.01482579688658 | 58.604837803300434 | 61.55637337503976 | 6745 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_thigh | 1 | 64.06226834692364 | 62.33163438330016 | 64.65674665674102 | 6745 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_thigh | 1 | 60.84507042253521 | 59.456751533278116 | 62.99469016964048 | 6745 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_thigh | 1 | 65.08524833209785 | 63.719183443439405 | 66.06190352667551 | 6745 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_thigh | 1 | 69.72572275759822 | 68.49516581113917 | 70.75356175828298 | 6745 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_thigh | 1 | 61.26019273535953 | 60.02040440089834 | 63.28682752900486 | 6745 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_thigh | 1 | 67.19051148999259 | 65.75816913440215 | 68.42295822496429 | 6745 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_thigh | 1 | 61.882876204595995 | 60.43719809457102 | 64.38675153747006 | 6745 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_thigh | 1 | 67.22016308376575 | 66.74131799459354 | 68.37425393622756 | 6745 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_thigh | 1 | 71.14899925871015 | 70.10944828802306 | 72.11425314848893 | 6745 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_thigh | 1 | 61.17123795404002 | 60.27431423491898 | 63.41807449210395 | 6745 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_thigh | 1 | 68.83617494440327 | 67.61037240273737 | 70.17283795042381 | 6745 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_thigh | 1 | 62.075611564121566 | 60.73497961039109 | 64.28457682561687 | 6745 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_thigh | 1 | 68.51000741289845 | 68.64048541411256 | 69.64874858375435 | 6745 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_thigh | 1 | 71.74203113417347 | 70.82015987941841 | 72.81545512713103 | 6745 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_thigh | 1 | 62.03113417346182 | 61.424350417259156 | 64.51576305325617 | 6745 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_thigh | 1 | 69.26612305411416 | 67.97465490624846 | 70.60886843852663 | 6745 | ok |
| halo | 0.789 | 1nn | 128 | True | True | phone_thigh | 1 | 62.082514734774065 | 56.9062601005217 | 66.34542934893535 | 6617 | ok |
| halo | 0.789 | prototype | 128 | True | True | phone_thigh | 1 | 68.11243766057126 | 66.32510879652591 | 70.22591270667219 | 6617 | ok |
| halo | 0.789 | ridge | 128 | True | True | phone_thigh | 1 | 71.37675683844643 | 67.05242926832152 | 73.83471227426026 | 6617 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | phone_thigh | 1 | 61.493123772102166 | 56.918257227996726 | 65.2008881855396 | 6617 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | phone_thigh | 1 | 69.62369653921718 | 65.73366479522804 | 72.14238962764881 | 6617 | ok |
| halo | 2.203 | halo-classifier | 0 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 54.35151515151515 | 59.69181276283662 | 64.54411815264585 | 4125 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| halo | 0.789 | 1nn | 1 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 62.787878787878796 | 59.99817408535924 | 64.37735634831114 | 4125 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 62.787878787878796 | 59.99817408535924 | 64.37735634831114 | 4125 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 63.030303030303024 | 60.41446593922891 | 65.55708195081223 | 4125 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 58.084848484848486 | 58.18620079470119 | 64.42222936934725 | 4125 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 62.787878787878796 | 60.21665710709776 | 64.79757549533186 | 4125 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 66.81212121212121 | 65.41988784450497 | 68.6383799092646 | 4125 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 63.10303030303031 | 59.98203064248301 | 65.84152786217844 | 4125 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 66.81212121212121 | 64.5635616972795 | 69.09040293384163 | 4125 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 63.07878787878788 | 65.16391048662756 | 69.17446101582463 | 4125 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 67.53939393939395 | 66.44369837230653 | 69.52984542861917 | 4125 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 70.88484848484849 | 70.05981686327266 | 71.98428647835829 | 4125 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 71.78181818181818 | 70.06473496253518 | 73.58944865638865 | 4125 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 73.45454545454545 | 71.89417729609337 | 74.70789220608019 | 4125 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 65.4060606060606 | 68.62061189836743 | 71.20256049969949 | 4125 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 73.30909090909091 | 72.61118969085585 | 74.23781809359556 | 4125 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 71.78181818181818 | 71.01826599412078 | 72.7437756056905 | 4125 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 74.23030303030303 | 72.94735083254587 | 75.3136846819612 | 4125 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 75.2969696969697 | 74.1447751234247 | 76.08402212326013 | 4125 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 67.85454545454546 | 70.56365155132657 | 72.51668367200962 | 4125 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 75.10303030303031 | 74.36902289910196 | 76.25052865131212 | 4125 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 72.48484848484848 | 71.52719040955402 | 73.42405532607287 | 4125 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 75.9030303030303 | 74.97660069170367 | 77.15158603193757 | 4125 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 77.86666666666666 | 76.65440657586846 | 78.69206718297137 | 4125 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 68.26666666666667 | 71.38821554884194 | 73.29236604227157 | 4125 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 76.29090909090908 | 75.59663974281192 | 77.22583534201424 | 4125 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 72.14545454545454 | 70.27435689541855 | 72.47133362484843 | 4125 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 76.0969696969697 | 75.2031617537937 | 77.20333518084611 | 4125 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 78.61818181818182 | 77.08431515398723 | 78.90860399317883 | 4125 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 67.95151515151515 | 70.98504948977482 | 72.80072396855613 | 4125 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 76.48484848484848 | 75.40559325805827 | 76.93511087515378 | 4125 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 72.30731225296442 | 58.55402155693214 | 68.77482484993854 | 4048 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 76.55632411067194 | 64.09891048550409 | 75.31175717419835 | 4048 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 79.2490118577075 | 66.65777020253972 | 77.3441705738459 | 4048 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 67.71245059288538 | 59.53531392812447 | 69.55480715810705 | 4048 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 76.60573122529645 | 63.85313664644229 | 74.37866666446308 | 4048 | ok |
| halo | 0.789 | all | 128 | True | True | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_waist | 1 | 32.518509813139026 | 25.236477876029927 | 35.12696160988996 | 17018 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_waist | 1 | 37.560230344341285 | 35.08125289862566 | 36.789256500993474 | 17018 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_waist | 1 | 37.560230344341285 | 35.08125289862566 | 36.789256500993474 | 17018 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_waist | 1 | 38.036196967916325 | 35.40105457019331 | 37.17437859451241 | 17018 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_waist | 1 | 42.67246444940651 | 39.962857399134535 | 42.04308127938396 | 17018 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_waist | 1 | 41.967328710776826 | 39.339053359782426 | 41.51692797454993 | 17018 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_waist | 1 | 42.90750969561641 | 39.7937752341233 | 42.09265621484812 | 17018 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_waist | 1 | 45.06404982959219 | 42.309987077239676 | 44.45417516899951 | 17018 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_waist | 1 | 44.793747796450816 | 42.05694729084682 | 44.42762750515394 | 17018 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_waist | 1 | 47.379245504759666 | 44.02670956815818 | 46.84502123327717 | 17018 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_waist | 1 | 48.360559407685976 | 45.590062360572716 | 47.615646488988105 | 17018 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_waist | 1 | 49.37713009754378 | 46.5648776989042 | 49.28537789303352 | 17018 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_waist | 1 | 53.14960629921261 | 50.00833777928374 | 52.99237770289362 | 17018 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_waist | 1 | 51.081208132565514 | 48.684314153544946 | 50.98234440944272 | 17018 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_waist | 1 | 52.93218944646845 | 50.17046644521217 | 52.989015921776215 | 17018 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_waist | 1 | 57.95628158420496 | 55.55778764680102 | 58.51932482000346 | 17018 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_waist | 1 | 53.76660007051357 | 51.78219201460203 | 54.16060975305358 | 17018 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_waist | 1 | 55.288518039722646 | 52.47574462170208 | 55.3085278581247 | 17018 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_waist | 1 | 62.31637090139852 | 61.000722247758254 | 63.029730883506765 | 17018 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_waist | 1 | 55.75860853214244 | 53.71103272315536 | 56.01162701044438 | 17018 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_waist | 1 | 56.46374427077212 | 53.86315209541132 | 56.638716663790746 | 17018 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_waist | 1 | 65.42484428252439 | 64.76245169814665 | 66.54184613670344 | 17018 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | phone_waist | 1 | 58.12668938770713 | 56.30974412332661 | 58.386060578018174 | 17018 | ok |
| harnet | 4.49 | prototype | 128 | False | False | phone_waist | 1 | 56.945587025502405 | 54.25221798432391 | 56.97795038775434 | 17018 | ok |
| harnet | 4.49 | ridge | 128 | False | False | phone_waist | 1 | 68.06322717123047 | 67.87090231146365 | 69.06542552606234 | 17018 | ok |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_forearm | 1 | 34.01613109717066 | 24.513892683285142 | 33.30204182908423 | 7811 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_forearm | 1 | 36.154141595186275 | 31.679422587267126 | 34.388724007083994 | 7811 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_forearm | 1 | 36.154141595186275 | 31.679422587267126 | 34.388724007083994 | 7811 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_forearm | 1 | 36.66623991806427 | 32.06898026155435 | 34.65296440721368 | 7811 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_forearm | 1 | 38.1385225963385 | 33.97787706763598 | 36.779152833433315 | 7811 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_forearm | 1 | 38.67622583536039 | 34.26699098994676 | 37.14241345566266 | 7811 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_forearm | 1 | 39.31634873895788 | 34.43071944976307 | 37.510395186510536 | 7811 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_forearm | 1 | 41.595186275764945 | 37.54778700633985 | 40.623824894217144 | 7811 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_forearm | 1 | 42.68339521188068 | 37.702334586641854 | 40.73381746505686 | 7811 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_forearm | 1 | 44.24529509665856 | 38.942426633996405 | 42.82570780951238 | 7811 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_forearm | 1 | 43.989245935219564 | 39.831854193742366 | 43.010307583887894 | 7811 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_forearm | 1 | 45.960824478299834 | 40.90380671969614 | 44.972950833631074 | 7811 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_forearm | 1 | 48.559723466905645 | 43.42718044099874 | 48.17729659756117 | 7811 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_forearm | 1 | 45.80719498143643 | 41.83162486066182 | 44.968966570616466 | 7811 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_forearm | 1 | 48.86698246063244 | 42.66610713024508 | 46.87000846605979 | 7811 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_forearm | 1 | 51.41467161695046 | 46.502746472563125 | 51.26261534925287 | 7811 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_forearm | 1 | 47.356292408142366 | 43.90292606794479 | 47.006769013447695 | 7811 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_forearm | 1 | 50.31366022276277 | 44.13911702265514 | 48.927870049359136 | 7811 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_forearm | 1 | 55.42184099347075 | 52.29581299905613 | 56.46499246051695 | 7811 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_forearm | 1 | 49.03341441556779 | 46.58432908874271 | 49.880809882586476 | 7811 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_forearm | 1 | 51.55549865574191 | 44.661454920974435 | 49.65378715615993 | 7811 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_forearm | 1 | 57.84150556906926 | 56.141999378049626 | 59.55817487289907 | 7811 | ok |
| harnet | 4.49 | all | 128 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_thigh | 1 | 32.40919199406968 | 26.49427402237407 | 38.2339947798223 | 6745 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_thigh | 1 | 35.507783543365456 | 33.532377748078815 | 36.72511547110051 | 6745 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_thigh | 1 | 35.507783543365456 | 33.532377748078815 | 36.72511547110051 | 6745 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_thigh | 1 | 36.18977020014826 | 34.376693067857396 | 37.80764730862856 | 6745 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_thigh | 1 | 38.53224610822832 | 37.11207779760778 | 40.84334376278911 | 6745 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_thigh | 1 | 38.16160118606375 | 36.50444244346414 | 40.130413186411786 | 6745 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_thigh | 1 | 39.747961452928095 | 38.330323691491294 | 42.56472107781361 | 6745 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_thigh | 1 | 42.74277242401779 | 41.51979860950459 | 45.27846579411235 | 6745 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_thigh | 1 | 42.52038547071905 | 41.786400946462585 | 45.93973139979955 | 6745 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_thigh | 1 | 44.996293550778354 | 44.34748394976455 | 49.20554840997235 | 6745 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_thigh | 1 | 45.974796145292814 | 45.15608888885539 | 49.07755010049763 | 6745 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_thigh | 1 | 45.277983691623426 | 45.2269617269779 | 49.68413615677729 | 6745 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_thigh | 1 | 49.26612305411416 | 49.1457627334879 | 54.83058947320535 | 6745 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_thigh | 1 | 47.9466271312083 | 46.873038783466015 | 50.400007517122134 | 6745 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_thigh | 1 | 47.487027427724236 | 47.39597311202572 | 51.4976062041723 | 6745 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_thigh | 1 | 52.37954040029652 | 52.31977355172555 | 57.62447359276122 | 6745 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_thigh | 1 | 50.80800593031876 | 50.10290268072708 | 53.351095841985554 | 6745 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_thigh | 1 | 48.70274277242402 | 48.947444723737135 | 53.946837471396925 | 6745 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_thigh | 1 | 55.58191252779837 | 55.021809252519475 | 60.69560645020869 | 6745 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_thigh | 1 | 52.33506300963677 | 51.86967430970779 | 54.99929393715469 | 6745 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_thigh | 1 | 49.25129725722758 | 49.68523651716668 | 54.520047610492824 | 6745 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_thigh | 1 | 57.39065974796145 | 57.38131476451282 | 61.964545329424624 | 6745 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | phone_thigh | 1 | 52.334894967507935 | 47.47963742673183 | 56.549054407538854 | 6617 | ok |
| harnet | 4.49 | prototype | 128 | False | False | phone_thigh | 1 | 49.40305274293487 | 45.74747440463266 | 56.20747286463457 | 6617 | ok |
| harnet | 4.49 | ridge | 128 | False | False | phone_thigh | 1 | 59.180897687773914 | 54.794656437517176 | 64.34825541296378 | 6617 | ok |
| harnet | 4.49 | zero_support | 0 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| harnet | 4.49 | 1nn | 1 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 44.266666666666666 | 40.2103238464801 | 44.278878936369416 | 4125 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 44.266666666666666 | 40.2103238464801 | 44.278878936369416 | 4125 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 43.56363636363636 | 39.318630396776285 | 43.978605405000266 | 4125 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 49.09090909090909 | 44.65161017979856 | 47.956395108939084 | 4125 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 46.27878787878788 | 42.674342990736555 | 47.238751489458885 | 4125 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 46.375757575757575 | 42.143780942390244 | 47.63361488973461 | 4125 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 52.193939393939395 | 48.863147162456755 | 51.56242550125422 | 4125 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 51.10303030303031 | 47.49977057617796 | 52.18484085476703 | 4125 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 52.21818181818182 | 48.73227550354111 | 53.954931243420276 | 4125 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 54.73939393939394 | 52.10299945999388 | 54.816684869097266 | 4125 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 53.35757575757576 | 50.19144235494366 | 55.206141781496676 | 4125 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 56.0969696969697 | 53.73400241586677 | 58.591358955324104 | 4125 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 56.67878787878789 | 54.7985861776521 | 56.799366473737614 | 4125 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 57.06666666666666 | 53.630113575330874 | 57.98324632963943 | 4125 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 60.484848484848484 | 59.13556796751184 | 63.411672508086724 | 4125 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 57.81818181818181 | 56.54154957156177 | 58.97436898710521 | 4125 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 58.060606060606055 | 55.352310918942734 | 60.002682988977774 | 4125 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 62.763636363636365 | 62.702825370972626 | 66.39790258262468 | 4125 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 60.42490118577075 | 51.72716398700552 | 59.782542653427285 | 4048 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 58.448616600790515 | 49.47967869252575 | 58.0924323544619 | 4048 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 64.30335968379447 | 54.77413415220453 | 65.22422727227458 | 4048 | ok |
| harnet | 4.49 | all | 128 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_waist | 1 | 38.31237513221295 | 32.441925767749545 | 42.58139408939303 | 17018 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_waist | 1 | 55.923140204489364 | 54.90864623544917 | 57.906044278018165 | 17018 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_waist | 1 | 55.923140204489364 | 54.90864623544917 | 57.906044278018165 | 17018 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_waist | 1 | 54.37184157950406 | 53.239460469304355 | 56.85611642431262 | 17018 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_waist | 1 | 62.13421083558585 | 62.05876464454973 | 64.33382802509396 | 17018 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_waist | 1 | 60.16570689857797 | 60.105232286676525 | 62.81593075926477 | 17018 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_waist | 1 | 59.39005758608532 | 58.08128834732037 | 62.03906752932789 | 17018 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_waist | 1 | 64.71970854389471 | 65.006712895238 | 66.88262235004171 | 17018 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_waist | 1 | 63.86179339522858 | 64.79372569795827 | 66.57806516977311 | 17018 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_waist | 1 | 62.768833000352565 | 62.366395548040956 | 65.46995052175359 | 17018 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_waist | 1 | 66.6529556939711 | 67.22185359423332 | 68.89705522458023 | 17018 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_waist | 1 | 66.75284992361028 | 68.03844600354418 | 69.2201695327696 | 17018 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_waist | 1 | 66.9761429075097 | 66.9626357979458 | 69.36564398166377 | 17018 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_waist | 1 | 67.95158067928077 | 68.36990425953049 | 70.16457008925572 | 17018 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_waist | 1 | 69.91420848513339 | 71.25937428959195 | 72.12296447496114 | 17018 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_waist | 1 | 70.23739569867199 | 69.92396780993765 | 72.45173030738327 | 17018 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_waist | 1 | 69.40298507462687 | 70.25268103444213 | 71.54843684420867 | 17018 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_waist | 1 | 71.67704783170761 | 72.99663395065244 | 73.80184159443306 | 17018 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_waist | 1 | 71.60065812668938 | 71.10718298771069 | 73.91050136383845 | 17018 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_waist | 1 | 69.69679163238924 | 70.6563818097615 | 71.88506592513708 | 17018 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_waist | 1 | 72.03549183217768 | 73.33236018008085 | 74.06604220290205 | 17018 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_waist | 1 | 73.02268186625925 | 72.90114487616715 | 75.33831381255061 | 17018 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | phone_waist | 1 | 70.26677635444824 | 71.61167679742553 | 72.52431746046723 | 17018 | ok |
| unimts | 68.61 | prototype | 128 | True | False | phone_waist | 1 | 72.28816547185333 | 73.65902204256727 | 74.39148444352186 | 17018 | ok |
| unimts | 68.61 | ridge | 128 | True | False | phone_waist | 1 | 73.92760606416735 | 73.87050198768796 | 76.13625668171767 | 17018 | ok |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_forearm | 1 | 49.48150044808603 | 35.802399776013246 | 47.54527991792773 | 7811 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_forearm | 1 | 46.204071181666876 | 43.84694624714459 | 48.51572127457605 | 7811 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_forearm | 1 | 46.204071181666876 | 43.84694624714459 | 48.51572127457605 | 7811 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_forearm | 1 | 43.59236973498912 | 41.364479785127685 | 47.260146541704394 | 7811 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_forearm | 1 | 50.49289463577007 | 48.90726273225074 | 53.183774874918385 | 7811 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_forearm | 1 | 48.3676865958264 | 46.6904033380123 | 50.80546741807483 | 7811 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_forearm | 1 | 46.306490846242475 | 44.142225664213775 | 49.45790527350209 | 7811 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_forearm | 1 | 53.47586736653438 | 52.199837688695716 | 56.85454076567902 | 7811 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_forearm | 1 | 51.74753552682115 | 51.13456912696317 | 54.73295973763847 | 7811 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_forearm | 1 | 50.569709384201765 | 49.07935662082495 | 53.98931774620542 | 7811 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_forearm | 1 | 54.78171808987325 | 53.65146015555706 | 58.37546480992985 | 7811 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_forearm | 1 | 52.13160926897964 | 52.107252619896435 | 55.35955665354491 | 7811 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_forearm | 1 | 52.170016643195495 | 51.326371928843095 | 55.50442771344414 | 7811 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_forearm | 1 | 55.60107540647804 | 54.990838811141394 | 59.36972113975803 | 7811 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_forearm | 1 | 53.73191652797337 | 53.13018043726706 | 56.407798257140584 | 7811 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_forearm | 1 | 54.43605172193061 | 52.93556759471443 | 57.42376234294719 | 7811 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_forearm | 1 | 56.343617974651124 | 56.32607487592228 | 60.74962495798913 | 7811 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_forearm | 1 | 56.02355652285239 | 55.064530238221245 | 58.544040909903764 | 7811 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_forearm | 1 | 56.86851875560107 | 54.60883295898413 | 59.58119867901846 | 7811 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_forearm | 1 | 57.162975291255925 | 57.555078473935836 | 61.621422253635295 | 7811 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_forearm | 1 | 56.10037127128409 | 54.349795248341195 | 58.4887187747221 | 7811 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_forearm | 1 | 58.05914735629241 | 55.332127152792424 | 60.68967564994055 | 7811 | ok |
| unimts | 68.61 | all | 128 | True | False | phone_forearm | 1 |  |  |  |  | n/a |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_thigh | 1 | 27.739065974796144 | 19.34743756936538 | 31.34631037717195 | 6745 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_thigh | 1 | 49.22164566345441 | 49.56734851622057 | 54.249145027716686 | 6745 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_thigh | 1 | 49.22164566345441 | 49.56734851622057 | 54.249145027716686 | 6745 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_thigh | 1 | 45.63380281690141 | 46.13762600998397 | 52.1506606041648 | 6745 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_thigh | 1 | 58.22090437361008 | 59.08135205960647 | 62.93442191970138 | 6745 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_thigh | 1 | 52.5722757598221 | 53.89209950177134 | 58.12295628815618 | 6745 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_thigh | 1 | 51.11934766493699 | 51.339924273939694 | 57.74585541843125 | 6745 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_thigh | 1 | 63.439584877687174 | 64.65275923272492 | 67.88503753475132 | 6745 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_thigh | 1 | 56.21942179392142 | 59.40948830771754 | 62.765019822214875 | 6745 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_thigh | 1 | 56.085989621942176 | 57.87662699391155 | 63.27526579635666 | 6745 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_thigh | 1 | 64.56634544106745 | 66.4124261912161 | 68.95132465080657 | 6745 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_thigh | 1 | 56.76797627872499 | 61.459645276379774 | 64.02096907221473 | 6745 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_thigh | 1 | 57.16827279466271 | 60.72509799220486 | 64.62391882965642 | 6745 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_thigh | 1 | 64.7887323943662 | 66.61710501132508 | 69.42090746897874 | 6745 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_thigh | 1 | 59.49592290585619 | 64.76818167750396 | 66.45371522980912 | 6745 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_thigh | 1 | 60.65233506300963 | 65.21960879847994 | 67.9959915444885 | 6745 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_thigh | 1 | 64.72942920681987 | 66.66210257196397 | 69.7064245021386 | 6745 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_thigh | 1 | 61.393624907338776 | 66.6280432600377 | 67.93326141140274 | 6745 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_thigh | 1 | 62.68346923647145 | 67.04770030328994 | 70.10503333161995 | 6745 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_thigh | 1 | 64.49221645663454 | 67.03950475644595 | 69.72143622641784 | 6745 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_thigh | 1 | 62.49073387694588 | 67.72176229749948 | 68.61979026003435 | 6745 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_thigh | 1 | 63.91401037805782 | 67.4387050733629 | 70.86921707617365 | 6745 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | phone_thigh | 1 | 63.533323258274145 | 62.32830976173592 | 70.1275016790853 | 6617 | ok |
| unimts | 68.61 | prototype | 128 | True | False | phone_thigh | 1 | 63.6239987909929 | 66.26100555486056 | 70.62898270898094 | 6617 | ok |
| unimts | 68.61 | ridge | 128 | True | False | phone_thigh | 1 | 64.75744294997733 | 66.63643003853296 | 73.02128856136149 | 6617 | ok |
| unimts | 68.61 | zero_support | 0 | True | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| unimts | 68.61 | 1nn | 1 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 55.27272727272727 | 52.47068508085195 | 57.226048027501264 | 4125 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 55.27272727272727 | 52.47068508085195 | 57.226048027501264 | 4125 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 53.28484848484848 | 50.4221096079177 | 56.71465750068205 | 4125 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 60.72727272727273 | 58.775121474993576 | 62.46161079018424 | 4125 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 57.45454545454546 | 56.58332769317717 | 60.1417365683813 | 4125 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 56.33939393939394 | 54.41325791872011 | 59.65501129557116 | 4125 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 64.31515151515151 | 62.84696180101197 | 65.52436292993487 | 4125 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 59.00606060606061 | 60.29482477487926 | 62.391905326224176 | 4125 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 59.32121212121212 | 60.379666443995426 | 63.17910134526164 | 4125 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 67.7090909090909 | 67.21122561793406 | 70.05702822415702 | 4125 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 59.68484848484849 | 62.23759091580264 | 64.07701307938525 | 4125 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 59.70909090909091 | 62.35289464272354 | 64.57277440233844 | 4125 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 68.4121212121212 | 68.04524107033994 | 70.34838824506292 | 4125 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 61.38181818181818 | 64.46841846626614 | 65.62590907131975 | 4125 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 63.32121212121212 | 66.22337706226784 | 67.49211693121902 | 4125 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 67.87878787878789 | 67.19518482400008 | 69.58881621189342 | 4125 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 63.87878787878788 | 66.38549509274854 | 67.30474166258657 | 4125 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 67.0060606060606 | 68.509970349326 | 69.85004346510905 | 4125 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 67.14426877470355 | 55.28389948549406 | 64.53505023795206 | 4048 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 64.550395256917 | 55.032211423344044 | 64.3133408314304 | 4048 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 68.08300395256917 | 56.77146351849943 | 66.83566062351723 | 4048 | ok |
| unimts | 68.61 | all | 128 | True | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_waist | 1 | 14.184980608767187 | 3.1057019349526556 | 12.5 | 17018 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_waist | 1 | 22.687742390410154 | 21.48252878455907 | 22.043819234013995 | 17018 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_waist | 1 | 22.687742390410154 | 21.48252878455907 | 22.043819234013995 | 17018 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_waist | 1 | 21.671171700552357 | 19.970269716872917 | 21.193495991592567 | 17018 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_waist | 1 | 24.074509343048536 | 22.930832741834422 | 24.2358560647125 | 17018 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_waist | 1 | 24.00399576918557 | 22.58192583410818 | 23.593308093883223 | 17018 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_waist | 1 | 23.016805735104008 | 20.8558749356984 | 22.47927055177929 | 17018 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_waist | 1 | 24.885415442472674 | 23.542172170090733 | 24.194275467119233 | 17018 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_waist | 1 | 23.80420730990716 | 22.07358028268559 | 22.922519695006375 | 17018 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_waist | 1 | 23.375249735574098 | 20.703044357648533 | 22.38357174568532 | 17018 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_waist | 1 | 26.965565871430254 | 25.518697661577033 | 26.12158544439601 | 17018 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_waist | 1 | 25.243859442942767 | 23.303102800638346 | 24.63390081294744 | 17018 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_waist | 1 | 24.550475966623576 | 21.556269163959264 | 23.956147356457677 | 17018 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_waist | 1 | 28.963450464214365 | 27.436471968592564 | 28.33780136908311 | 17018 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_waist | 1 | 25.790339640380772 | 23.413691260926484 | 24.847716914582538 | 17018 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_waist | 1 | 25.426019508755438 | 21.63511879337116 | 24.554675557845957 | 17018 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_waist | 1 | 30.15042895757433 | 28.64250042505957 | 29.80026056392027 | 17018 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_waist | 1 | 26.501351510165705 | 23.502950901940505 | 25.38242514591571 | 17018 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_waist | 1 | 26.712892231754616 | 22.459693713012957 | 25.940429132361032 | 17018 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_waist | 1 | 32.1306851568927 | 30.39371783172427 | 31.373602235044107 | 17018 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_waist | 1 | 26.560112821718178 | 23.077954657352606 | 25.215180596118014 | 17018 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_waist | 1 | 28.158420495945467 | 23.54772262629573 | 27.291557498538715 | 17018 | ok |
| normwear | 1293.86 | 1nn | 128 | True | False | phone_waist | 1 | 33.88177224115642 | 32.10611485422322 | 33.46006308562002 | 17018 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | phone_waist | 1 | 26.48959924785521 | 22.550468498879788 | 25.132577068744173 | 17018 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | phone_waist | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_forearm | 1 | 16.412751248239662 | 3.524689321456065 | 12.5 | 7811 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_forearm | 1 | 21.725771348098835 | 19.590541441891467 | 21.24414253940719 | 7811 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_forearm | 1 | 21.725771348098835 | 19.590541441891467 | 21.24414253940719 | 7811 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_forearm | 1 | 18.281910126744336 | 15.822047474982542 | 18.705977877520784 | 7811 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_forearm | 1 | 23.64614005889131 | 20.879017551616407 | 21.81481623102655 | 7811 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_forearm | 1 | 21.49532710280374 | 19.05116441779177 | 20.931102822889002 | 7811 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_forearm | 1 | 18.883625656125975 | 15.990500479692077 | 19.357810795754798 | 7811 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_forearm | 1 | 26.757137370375112 | 23.593222764736698 | 24.31860309661225 | 7811 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_forearm | 1 | 22.289079503264627 | 19.50230908057652 | 21.679867860098007 | 7811 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_forearm | 1 | 19.139674817564973 | 15.851156507077329 | 19.717834944772044 | 7811 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_forearm | 1 | 28.767123287671232 | 25.40668468714957 | 26.922764233886998 | 7811 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_forearm | 1 | 20.5351427474075 | 17.654551798234795 | 20.313082895061328 | 7811 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_forearm | 1 | 18.64037895275893 | 14.936561077812573 | 19.09776037484649 | 7811 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_forearm | 1 | 31.08436819869415 | 27.05325780912458 | 28.28148099995198 | 7811 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_forearm | 1 | 20.20227883753681 | 17.26305607695795 | 20.94901551999491 | 7811 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_forearm | 1 | 18.819613365766227 | 14.453566419773702 | 19.47055685234594 | 7811 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_forearm | 1 | 33.33760081935731 | 29.214480805998523 | 31.063860824244816 | 7811 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_forearm | 1 | 20.03584688260146 | 16.75993840239588 | 20.881792271960776 | 7811 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_forearm | 1 | 20.03584688260146 | 15.50231511350649 | 20.733338621173008 | 7811 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_forearm | 1 | 33.81129176801946 | 29.389040631320107 | 31.052512279362936 | 7811 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_forearm | 1 | 19.16527973370887 | 15.483277676128537 | 19.982278462063203 | 7811 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_forearm | 1 | 22.557931122775575 | 17.889173236286204 | 23.356422335529466 | 7811 | ok |
| normwear | 1293.86 | all | 128 | True | False | phone_forearm | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_thigh | 1 | 19.05114899925871 | 4.000622665006227 | 12.5 | 6745 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_thigh | 1 | 25.337286879169756 | 22.485884585449732 | 24.586832127997997 | 6745 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_thigh | 1 | 25.337286879169756 | 22.485884585449732 | 24.586832127997997 | 6745 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_thigh | 1 | 23.054114158636025 | 19.723184351314003 | 21.21279271008536 | 6745 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_thigh | 1 | 25.945144551519643 | 22.99531108533056 | 25.097364981075508 | 6745 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_thigh | 1 | 26.523350630096367 | 22.923981115916337 | 24.787326441525614 | 6745 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_thigh | 1 | 25.114899925871015 | 20.626925983445226 | 22.290096134115892 | 6745 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_thigh | 1 | 28.999258710155672 | 25.886302243149057 | 28.242777588582324 | 6745 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_thigh | 1 | 28.450704225352112 | 24.83840808891575 | 27.300661449265817 | 6745 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_thigh | 1 | 25.76723498888065 | 21.285357323931024 | 24.228617551088917 | 6745 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_thigh | 1 | 32.55744996293551 | 29.16706007475903 | 31.656328167403146 | 6745 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_thigh | 1 | 27.11638250555967 | 23.247965721242807 | 25.48858976244655 | 6745 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_thigh | 1 | 25.277983691623422 | 20.335881597776698 | 24.03016131782996 | 6745 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_thigh | 1 | 32.23128243143069 | 28.98500652726491 | 31.543891341907788 | 6745 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_thigh | 1 | 28.139362490733877 | 24.285906525620998 | 27.27938605876295 | 6745 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_thigh | 1 | 26.271312083024462 | 21.255669697949635 | 26.50638382389675 | 6745 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_thigh | 1 | 35.61156412157153 | 32.37705203379158 | 35.122615004989655 | 6745 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_thigh | 1 | 28.53965900667161 | 24.023301951302663 | 26.963438926089566 | 6745 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_thigh | 1 | 27.10155670867309 | 21.718612985608825 | 28.97259448724424 | 6745 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_thigh | 1 | 36.87175685693106 | 33.690387927093255 | 36.56817764657786 | 6745 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_thigh | 1 | 27.709414381022977 | 22.780039742710294 | 25.935075904117642 | 6745 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_thigh | 1 | 28.510007412898442 | 22.514404616130232 | 31.250137918374765 | 6745 | ok |
| normwear | 1293.86 | 1nn | 128 | True | False | phone_thigh | 1 | 37.55478313435091 | 32.63800210324451 | 38.95857322930247 | 6617 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | phone_thigh | 1 | 28.9103823484963 | 21.90086349815539 | 28.052582618553867 | 6617 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | phone_thigh | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | zero_support | 0 | True | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 25.13939393939394 | 21.395821224632787 | 22.879311171068068 | 4125 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 25.13939393939394 | 21.395821224632787 | 22.879311171068068 | 4125 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 23.78181818181818 | 19.719674188484976 | 22.33067013768352 | 4125 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 26.254545454545454 | 23.245866364530535 | 25.6954687920598 | 4125 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 26.254545454545454 | 23.1209859340964 | 25.852082384859443 | 4125 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 23.66060606060606 | 19.794022381564183 | 23.409608457578443 | 4125 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 26.400000000000002 | 23.454928458982373 | 26.126311957746307 | 4125 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 25.745454545454542 | 22.09316643893376 | 24.63580003919872 | 4125 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 24.46060606060606 | 19.983543980751644 | 23.5995588117916 | 4125 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 28.8969696969697 | 25.669420984376035 | 28.49337813217014 | 4125 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 26.884848484848483 | 22.659901900416052 | 25.06626097193574 | 4125 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 25.76969696969697 | 20.50322960075017 | 25.371623832673258 | 4125 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 31.272727272727273 | 27.88491021433557 | 31.31624680142402 | 4125 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 27.054545454545455 | 22.77392555756488 | 26.297730561199582 | 4125 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 26.836363636363636 | 21.274014694157596 | 27.89422499593208 | 4125 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 33.6 | 30.234312646039257 | 34.20116409985561 | 4125 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 27.515151515151516 | 22.99567253051162 | 27.04534699266169 | 4125 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 27.975757575757576 | 21.813966545933496 | 30.676685190686502 | 4125 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 34.708498023715414 | 28.796643222461917 | 33.255281767887915 | 4048 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 25.8399209486166 | 20.55110630562384 | 26.174000126192958 | 4048 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 28.705533596837945 | 21.419234947106446 | 32.34095836343084 | 4048 | ok |
| normwear | 1293.86 | all | 128 | True | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 1 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 2 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 4 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 8 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 16 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 32 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 64 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 128 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 1 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 2 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 4 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 8 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 16 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 32 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 64 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 128 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 1 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 2 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 4 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 8 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 16 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 32 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 64 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 128 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | zero_support | 0 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 1 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 2 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 4 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 8 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 16 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 32 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 64 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 128 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |

### shoaib
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | stream | n_devices | accuracy | f1_macro | balanced_accuracy | n_queries | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halo | 2.203 | halo-classifier | 0 | True | True | phone_right_pocket | 1 | 67.74603174603175 | 66.16407360173066 | 67.74603174603175 | 3150 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_right_pocket | 1 | 54.095238095238095 | 50.49920799637864 | 54.095238095238095 | 3150 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_right_pocket | 1 | 68.6031746031746 | 69.15273216752689 | 68.60317460317461 | 3150 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_right_pocket | 1 | 68.6031746031746 | 69.15273216752689 | 68.60317460317461 | 3150 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_right_pocket | 1 | 70.82539682539682 | 70.79683416740475 | 70.82539682539684 | 3150 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_right_pocket | 1 | 75.5873015873016 | 75.39853874598973 | 75.5873015873016 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_right_pocket | 1 | 69.17460317460318 | 69.44741269053407 | 69.17460317460316 | 3150 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_right_pocket | 1 | 76.73015873015872 | 77.01669417794189 | 76.73015873015872 | 3150 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_right_pocket | 1 | 77.68253968253968 | 77.94424009244946 | 77.68253968253968 | 3150 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_right_pocket | 1 | 78.44444444444446 | 78.11543587985248 | 78.44444444444446 | 3150 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_right_pocket | 1 | 80.25396825396825 | 79.83141765237472 | 80.25396825396825 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_right_pocket | 1 | 77.23809523809524 | 77.36877796512977 | 77.23809523809523 | 3150 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_right_pocket | 1 | 81.2063492063492 | 81.324975933741 | 81.20634920634922 | 3150 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_right_pocket | 1 | 82.44444444444444 | 82.55286552541897 | 82.44444444444444 | 3150 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_right_pocket | 1 | 83.23809523809523 | 82.92774863026946 | 83.23809523809524 | 3150 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_right_pocket | 1 | 81.96825396825396 | 81.61000375683992 | 81.96825396825396 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_right_pocket | 1 | 82.03174603174604 | 82.00520985104818 | 82.03174603174602 | 3150 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_right_pocket | 1 | 83.87301587301587 | 83.9271946243312 | 83.87301587301587 | 3150 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_right_pocket | 1 | 84.53968253968253 | 84.6425535804595 | 84.53968253968253 | 3150 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_right_pocket | 1 | 87.26984126984128 | 87.06682614884735 | 87.26984126984127 | 3150 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_right_pocket | 1 | 83.74603174603175 | 83.3751818266209 | 83.74603174603176 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_right_pocket | 1 | 84.98412698412699 | 84.92609209997852 | 84.98412698412697 | 3150 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_right_pocket | 1 | 86.44444444444444 | 86.41977099249367 | 86.44444444444444 | 3150 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_right_pocket | 1 | 86.12698412698413 | 86.25041937293864 | 86.12698412698413 | 3150 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_right_pocket | 1 | 89.55555555555556 | 89.47938184244137 | 89.55555555555557 | 3150 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_right_pocket | 1 | 84.98412698412699 | 84.76609059322635 | 84.984126984127 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_right_pocket | 1 | 86.57142857142858 | 86.50675752329342 | 86.57142857142858 | 3150 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_right_pocket | 1 | 88.47619047619048 | 88.44330830651701 | 88.47619047619048 | 3150 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_right_pocket | 1 | 86.57142857142858 | 86.71558603368526 | 86.57142857142858 | 3150 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_right_pocket | 1 | 91.61904761904762 | 91.57374513306061 | 91.6190476190476 | 3150 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_right_pocket | 1 | 84.98412698412699 | 84.73714373494896 | 84.98412698412699 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_right_pocket | 1 | 88.34920634920634 | 88.26863698020657 | 88.34920634920636 | 3150 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_right_pocket | 1 | 89.4920634920635 | 89.4528740035432 | 89.4920634920635 | 3150 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_right_pocket | 1 | 86.76190476190476 | 86.8601821672167 | 86.76190476190476 | 3150 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_right_pocket | 1 | 92.76190476190476 | 92.7156919685792 | 92.76190476190476 | 3150 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_right_pocket | 1 | 85.90476190476191 | 85.70689789350617 | 85.90476190476191 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_right_pocket | 1 | 88.82539682539684 | 88.76260507237681 | 88.82539682539684 | 3150 | ok |
| halo | 0.789 | 1nn | 128 | True | True | phone_right_pocket | 1 | 89.80952380952381 | 89.71751870948759 | 89.80952380952381 | 3150 | ok |
| halo | 0.789 | prototype | 128 | True | True | phone_right_pocket | 1 | 87.07936507936508 | 87.20886536979407 | 87.07936507936508 | 3150 | ok |
| halo | 0.789 | ridge | 128 | True | True | phone_right_pocket | 1 | 94.03174603174604 | 93.99137348261765 | 94.03174603174604 | 3150 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | phone_right_pocket | 1 | 86.09523809523809 | 85.94620632583863 | 86.0952380952381 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | phone_right_pocket | 1 | 89.17460317460318 | 89.08260045104477 | 89.17460317460318 | 3150 | ok |
| halo | 2.203 | halo-classifier | 0 | True | True | phone_left_pocket | 1 | 73.2063492063492 | 72.030213584507 | 73.2063492063492 | 3150 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_left_pocket | 1 | 58.0952380952381 | 55.82593783457405 | 58.09523809523809 | 3150 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_left_pocket | 1 | 71.26984126984127 | 71.16860925239462 | 71.26984126984127 | 3150 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_left_pocket | 1 | 71.26984126984127 | 71.16860925239462 | 71.26984126984127 | 3150 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_left_pocket | 1 | 73.07936507936508 | 72.55602546450775 | 73.07936507936509 | 3150 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_left_pocket | 1 | 77.52380952380953 | 76.62986344674565 | 77.52380952380952 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_left_pocket | 1 | 71.04761904761905 | 70.64585949164056 | 71.04761904761904 | 3150 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_left_pocket | 1 | 74.98412698412699 | 74.94017531190063 | 74.98412698412699 | 3150 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_left_pocket | 1 | 77.26984126984127 | 77.09062491322284 | 77.26984126984125 | 3150 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_left_pocket | 1 | 78.38095238095238 | 77.72907288556716 | 78.38095238095238 | 3150 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_left_pocket | 1 | 80.06349206349206 | 79.26880805169996 | 80.06349206349206 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_left_pocket | 1 | 76.0 | 75.68950288098574 | 76.0 | 3150 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_left_pocket | 1 | 80.76190476190476 | 80.58239793428699 | 80.76190476190477 | 3150 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_left_pocket | 1 | 82.5079365079365 | 82.22736330731662 | 82.50793650793649 | 3150 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_left_pocket | 1 | 84.25396825396825 | 83.49106594591912 | 84.25396825396825 | 3150 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_left_pocket | 1 | 82.57142857142857 | 81.7730630748366 | 82.57142857142858 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_left_pocket | 1 | 81.33333333333333 | 80.91933829402504 | 81.33333333333333 | 3150 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_left_pocket | 1 | 83.68253968253968 | 83.52047734138269 | 83.6825396825397 | 3150 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_left_pocket | 1 | 84.95238095238096 | 84.79316942560595 | 84.95238095238095 | 3150 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_left_pocket | 1 | 86.63492063492063 | 86.19340101934988 | 86.63492063492065 | 3150 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_left_pocket | 1 | 84.03174603174604 | 83.42098524400767 | 84.03174603174605 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_left_pocket | 1 | 84.47619047619047 | 84.04149956401541 | 84.47619047619048 | 3150 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_left_pocket | 1 | 85.80952380952381 | 85.74180068297659 | 85.80952380952381 | 3150 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_left_pocket | 1 | 86.47619047619047 | 86.3647729815653 | 86.47619047619048 | 3150 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_left_pocket | 1 | 89.65079365079364 | 89.4545403709205 | 89.65079365079366 | 3150 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_left_pocket | 1 | 84.92063492063492 | 84.38875245050092 | 84.92063492063492 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_left_pocket | 1 | 86.44444444444444 | 86.17842444040448 | 86.44444444444444 | 3150 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_left_pocket | 1 | 88.0 | 87.90111015422019 | 87.99999999999999 | 3150 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_left_pocket | 1 | 87.1111111111111 | 86.9972558324107 | 87.11111111111111 | 3150 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_left_pocket | 1 | 91.65079365079364 | 91.55014198965247 | 91.65079365079364 | 3150 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_left_pocket | 1 | 85.65079365079366 | 85.04851783084175 | 85.65079365079366 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_left_pocket | 1 | 88.38095238095238 | 88.11493936601634 | 88.38095238095238 | 3150 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_left_pocket | 1 | 89.87301587301587 | 89.81490388350932 | 89.87301587301586 | 3150 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_left_pocket | 1 | 87.3015873015873 | 87.22411537583208 | 87.30158730158732 | 3150 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_left_pocket | 1 | 92.76190476190476 | 92.69689857146709 | 92.76190476190474 | 3150 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_left_pocket | 1 | 86.19047619047619 | 85.6876046004805 | 86.19047619047619 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_left_pocket | 1 | 89.20634920634922 | 88.99825419558748 | 89.20634920634919 | 3150 | ok |
| halo | 0.789 | 1nn | 128 | True | True | phone_left_pocket | 1 | 91.01587301587301 | 90.97451367012323 | 91.01587301587301 | 3150 | ok |
| halo | 0.789 | prototype | 128 | True | True | phone_left_pocket | 1 | 87.65079365079364 | 87.57239351226652 | 87.65079365079366 | 3150 | ok |
| halo | 0.789 | ridge | 128 | True | True | phone_left_pocket | 1 | 94.22222222222221 | 94.19052894654129 | 94.22222222222221 | 3150 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | phone_left_pocket | 1 | 86.28571428571429 | 85.831222404646 | 86.28571428571429 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | phone_left_pocket | 1 | 89.46031746031747 | 89.2542588433829 | 89.46031746031747 | 3150 | ok |
| halo | 2.203 | halo-classifier | 0 | True | True | phone_belt | 1 | 74.34920634920636 | 74.14572351120626 | 74.34920634920636 | 3150 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_belt | 1 | 53.714285714285715 | 50.23882644602341 | 53.714285714285715 | 3150 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_belt | 1 | 78.12698412698413 | 78.10220824957295 | 78.12698412698413 | 3150 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_belt | 1 | 78.12698412698413 | 78.10220824957295 | 78.12698412698413 | 3150 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_belt | 1 | 79.26984126984127 | 78.92994084924486 | 79.26984126984127 | 3150 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_belt | 1 | 84.92063492063492 | 84.97018273760689 | 84.92063492063492 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_belt | 1 | 77.4920634920635 | 77.16392880333629 | 77.4920634920635 | 3150 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_belt | 1 | 83.42857142857143 | 83.38719315417282 | 83.42857142857143 | 3150 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_belt | 1 | 85.80952380952381 | 85.78225739259439 | 85.8095238095238 | 3150 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_belt | 1 | 85.80952380952381 | 85.49228704796768 | 85.80952380952381 | 3150 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_belt | 1 | 87.68253968253968 | 87.73997800666234 | 87.68253968253968 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_belt | 1 | 83.96825396825398 | 83.72152029812862 | 83.96825396825396 | 3150 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_belt | 1 | 86.5079365079365 | 86.55648384940571 | 86.5079365079365 | 3150 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_belt | 1 | 89.39682539682539 | 89.4370046024051 | 89.39682539682539 | 3150 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_belt | 1 | 89.55555555555556 | 89.44951487431234 | 89.55555555555554 | 3150 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_belt | 1 | 88.7936507936508 | 88.86514790423469 | 88.7936507936508 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_belt | 1 | 88.09523809523809 | 88.00181159963284 | 88.09523809523812 | 3150 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_belt | 1 | 88.98412698412699 | 89.04190607626855 | 88.98412698412699 | 3150 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_belt | 1 | 91.39682539682539 | 91.43045231858228 | 91.39682539682539 | 3150 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_belt | 1 | 91.07936507936508 | 91.03391369918539 | 91.07936507936508 | 3150 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_belt | 1 | 89.93650793650794 | 90.02480004906928 | 89.93650793650792 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_belt | 1 | 90.06349206349206 | 90.02058449352486 | 90.06349206349206 | 3150 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_belt | 1 | 90.25396825396825 | 90.32179367386419 | 90.25396825396824 | 3150 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_belt | 1 | 92.0 | 92.03483155863535 | 91.99999999999999 | 3150 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_belt | 1 | 92.88888888888889 | 92.87511275607618 | 92.8888888888889 | 3150 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_belt | 1 | 90.63492063492063 | 90.7111995040838 | 90.63492063492063 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_belt | 1 | 91.65079365079364 | 91.6365616379274 | 91.65079365079364 | 3150 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_belt | 1 | 90.66666666666666 | 90.6980139218054 | 90.66666666666666 | 3150 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_belt | 1 | 91.65079365079364 | 91.68796373779583 | 91.65079365079364 | 3150 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_belt | 1 | 93.04761904761905 | 93.06429386094719 | 93.04761904761904 | 3150 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_belt | 1 | 90.5079365079365 | 90.58476149618716 | 90.50793650793649 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_belt | 1 | 91.80952380952381 | 91.80048695083521 | 91.80952380952381 | 3150 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_belt | 1 | 91.68253968253968 | 91.71529844074234 | 91.68253968253967 | 3150 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_belt | 1 | 92.47619047619048 | 92.52187385683425 | 92.47619047619048 | 3150 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_belt | 1 | 94.12698412698413 | 94.13355098547163 | 94.12698412698413 | 3150 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_belt | 1 | 90.4126984126984 | 90.51393102359296 | 90.4126984126984 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_belt | 1 | 92.47619047619048 | 92.46289152798927 | 92.47619047619047 | 3150 | ok |
| halo | 0.789 | 1nn | 128 | True | True | phone_belt | 1 | 92.12698412698413 | 92.14180891799806 | 92.12698412698413 | 3150 | ok |
| halo | 0.789 | prototype | 128 | True | True | phone_belt | 1 | 92.44444444444444 | 92.4874438633656 | 92.44444444444444 | 3150 | ok |
| halo | 0.789 | ridge | 128 | True | True | phone_belt | 1 | 94.66666666666667 | 94.67868670419708 | 94.66666666666666 | 3150 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | phone_belt | 1 | 90.92063492063492 | 91.00836548068838 | 90.92063492063492 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | phone_belt | 1 | 92.88888888888889 | 92.89476488641102 | 92.88888888888887 | 3150 | ok |
| halo | 2.203 | halo-classifier | 0 | True | True | watch_wrist_proxy | 1 | 60.63492063492063 | 58.26896250766487 | 60.63492063492063 | 3150 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | watch_wrist_proxy | 1 | 41.460317460317455 | 33.47940963691723 | 41.46031746031747 | 3150 | ok |
| halo | 0.789 | 1nn | 1 | True | True | watch_wrist_proxy | 1 | 77.46031746031747 | 77.38587404360187 | 77.46031746031747 | 3150 | ok |
| halo | 0.789 | prototype | 1 | True | True | watch_wrist_proxy | 1 | 77.46031746031747 | 77.38587404360187 | 77.46031746031747 | 3150 | ok |
| halo | 0.789 | ridge | 1 | True | True | watch_wrist_proxy | 1 | 77.55555555555556 | 77.32797273802288 | 77.55555555555554 | 3150 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | watch_wrist_proxy | 1 | 79.36507936507937 | 79.13917370547783 | 79.36507936507937 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | watch_wrist_proxy | 1 | 76.85714285714286 | 76.65557860362725 | 76.85714285714286 | 3150 | ok |
| halo | 0.789 | 1nn | 2 | True | True | watch_wrist_proxy | 1 | 80.63492063492063 | 80.52270105107556 | 80.63492063492065 | 3150 | ok |
| halo | 0.789 | prototype | 2 | True | True | watch_wrist_proxy | 1 | 81.84126984126983 | 81.73629718657487 | 81.84126984126983 | 3150 | ok |
| halo | 0.789 | ridge | 2 | True | True | watch_wrist_proxy | 1 | 82.28571428571428 | 81.93579754053711 | 82.28571428571428 | 3150 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | watch_wrist_proxy | 1 | 82.34920634920636 | 82.16041734687192 | 82.34920634920634 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | watch_wrist_proxy | 1 | 80.92063492063491 | 80.76886930025876 | 80.92063492063491 | 3150 | ok |
| halo | 0.789 | 1nn | 4 | True | True | watch_wrist_proxy | 1 | 81.55555555555556 | 81.43853316686923 | 81.55555555555554 | 3150 | ok |
| halo | 0.789 | prototype | 4 | True | True | watch_wrist_proxy | 1 | 84.57142857142857 | 84.32523581911708 | 84.57142857142856 | 3150 | ok |
| halo | 0.789 | ridge | 4 | True | True | watch_wrist_proxy | 1 | 84.66666666666667 | 84.20104816837778 | 84.66666666666667 | 3150 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | watch_wrist_proxy | 1 | 83.23809523809523 | 83.04689944362434 | 83.23809523809526 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | watch_wrist_proxy | 1 | 82.98412698412699 | 82.68647926823915 | 82.98412698412699 | 3150 | ok |
| halo | 0.789 | 1nn | 8 | True | True | watch_wrist_proxy | 1 | 83.07936507936508 | 82.88340651786473 | 83.07936507936508 | 3150 | ok |
| halo | 0.789 | prototype | 8 | True | True | watch_wrist_proxy | 1 | 86.92063492063492 | 86.69907867494061 | 86.92063492063492 | 3150 | ok |
| halo | 0.789 | ridge | 8 | True | True | watch_wrist_proxy | 1 | 86.44444444444444 | 85.98864159010908 | 86.44444444444446 | 3150 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | watch_wrist_proxy | 1 | 85.23809523809524 | 85.06618142029564 | 85.23809523809523 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | watch_wrist_proxy | 1 | 85.04761904761905 | 84.7593040225219 | 85.04761904761905 | 3150 | ok |
| halo | 0.789 | 1nn | 16 | True | True | watch_wrist_proxy | 1 | 83.68253968253968 | 83.6651854283361 | 83.6825396825397 | 3150 | ok |
| halo | 0.789 | prototype | 16 | True | True | watch_wrist_proxy | 1 | 87.20634920634922 | 86.95873840341312 | 87.20634920634922 | 3150 | ok |
| halo | 0.789 | ridge | 16 | True | True | watch_wrist_proxy | 1 | 87.36507936507937 | 86.96351614332426 | 87.36507936507937 | 3150 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | watch_wrist_proxy | 1 | 85.23809523809524 | 85.28048751213258 | 85.23809523809524 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | watch_wrist_proxy | 1 | 86.22222222222223 | 86.0180283452632 | 86.22222222222223 | 3150 | ok |
| halo | 0.789 | 1nn | 32 | True | True | watch_wrist_proxy | 1 | 83.80952380952381 | 83.67546418385486 | 83.80952380952381 | 3150 | ok |
| halo | 0.789 | prototype | 32 | True | True | watch_wrist_proxy | 1 | 87.20634920634922 | 86.91372899322661 | 87.2063492063492 | 3150 | ok |
| halo | 0.789 | ridge | 32 | True | True | watch_wrist_proxy | 1 | 87.68253968253968 | 87.37368351041309 | 87.68253968253967 | 3150 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | watch_wrist_proxy | 1 | 84.92063492063492 | 84.80705549810304 | 84.92063492063491 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | watch_wrist_proxy | 1 | 86.12698412698413 | 85.78200282825189 | 86.12698412698413 | 3150 | ok |
| halo | 0.789 | 1nn | 64 | True | True | watch_wrist_proxy | 1 | 84.73015873015873 | 84.69350422606774 | 84.73015873015873 | 3150 | ok |
| halo | 0.789 | prototype | 64 | True | True | watch_wrist_proxy | 1 | 87.5873015873016 | 87.25591726567103 | 87.5873015873016 | 3150 | ok |
| halo | 0.789 | ridge | 64 | True | True | watch_wrist_proxy | 1 | 89.04761904761904 | 88.91676303779049 | 89.04761904761905 | 3150 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | watch_wrist_proxy | 1 | 85.4920634920635 | 85.4710114503801 | 85.4920634920635 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | watch_wrist_proxy | 1 | 86.57142857142858 | 86.31156895239715 | 86.57142857142858 | 3150 | ok |
| halo | 0.789 | 1nn | 128 | True | True | watch_wrist_proxy | 1 | 85.17460317460316 | 85.18763832297758 | 85.17460317460318 | 3150 | ok |
| halo | 0.789 | prototype | 128 | True | True | watch_wrist_proxy | 1 | 87.84126984126985 | 87.55169907084117 | 87.84126984126985 | 3150 | ok |
| halo | 0.789 | ridge | 128 | True | True | watch_wrist_proxy | 1 | 89.33333333333333 | 89.22215837495925 | 89.33333333333333 | 3150 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | watch_wrist_proxy | 1 | 85.42857142857143 | 85.3818851242883 | 85.42857142857143 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | watch_wrist_proxy | 1 | 86.8888888888889 | 86.64321249392967 | 86.88888888888889 | 3150 | ok |
| halo | 2.203 | halo-classifier | 0 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 74.28571428571429 | 71.62018982007791 | 74.28571428571429 | 3150 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| halo | 0.789 | 1nn | 1 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 82.47619047619048 | 82.59240582138406 | 82.47619047619048 | 3150 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 82.47619047619048 | 82.59240582138406 | 82.47619047619048 | 3150 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 83.93650793650794 | 83.80937572272369 | 83.93650793650794 | 3150 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 84.98412698412699 | 84.66024604401683 | 84.98412698412697 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 82.76190476190476 | 82.69202235968874 | 82.76190476190476 | 3150 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 86.60317460317461 | 86.6405758857217 | 86.60317460317461 | 3150 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 87.20634920634922 | 87.25118816572787 | 87.2063492063492 | 3150 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 88.63492063492063 | 88.49341360430805 | 88.63492063492063 | 3150 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 87.5873015873016 | 87.3659517967643 | 87.5873015873016 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 87.14285714285714 | 87.11051491166681 | 87.14285714285714 | 3150 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 88.7936507936508 | 88.80783339164768 | 88.7936507936508 | 3150 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 91.07936507936508 | 91.09473089352458 | 91.07936507936508 | 3150 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 92.06349206349206 | 91.97609086451799 | 92.06349206349206 | 3150 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 89.04761904761904 | 88.97782100832094 | 89.04761904761904 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 90.47619047619048 | 90.44933347907032 | 90.47619047619048 | 3150 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 91.9047619047619 | 91.87384207562145 | 91.90476190476191 | 3150 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 92.19047619047619 | 92.18638621280951 | 92.19047619047619 | 3150 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 93.71428571428572 | 93.64486270094174 | 93.71428571428571 | 3150 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 90.95238095238095 | 90.94713880286228 | 90.95238095238095 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 92.28571428571428 | 92.22905081517202 | 92.28571428571428 | 3150 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 93.61904761904762 | 93.59775383904024 | 93.6190476190476 | 3150 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 93.14285714285714 | 93.14608790176605 | 93.14285714285714 | 3150 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 95.58730158730158 | 95.56989390534338 | 95.58730158730158 | 3150 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 91.9047619047619 | 91.95685423844354 | 91.90476190476191 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 94.0 | 93.94970428525777 | 94.0 | 3150 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 94.06349206349206 | 94.05613622224311 | 94.06349206349206 | 3150 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 93.4920634920635 | 93.48335799938654 | 93.4920634920635 | 3150 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 95.84126984126983 | 95.83414220240162 | 95.84126984126983 | 3150 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 91.93650793650794 | 91.96288783021149 | 91.93650793650792 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 94.63492063492063 | 94.59119094493695 | 94.63492063492063 | 3150 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 95.11111111111111 | 95.10557985125747 | 95.11111111111113 | 3150 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 93.77777777777779 | 93.77441800728849 | 93.77777777777777 | 3150 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 97.14285714285714 | 97.13612618812233 | 97.14285714285715 | 3150 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 92.0952380952381 | 92.12184947908352 | 92.09523809523807 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 94.88888888888889 | 94.8489682720359 | 94.88888888888889 | 3150 | ok |
| halo | 0.789 | 1nn | 128 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 95.11111111111111 | 95.09242245029313 | 95.11111111111109 | 3150 | ok |
| halo | 0.789 | prototype | 128 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 93.61904761904762 | 93.62351348897334 | 93.61904761904762 | 3150 | ok |
| halo | 0.789 | ridge | 128 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 97.4920634920635 | 97.49096332022773 | 97.4920634920635 | 3150 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 91.93650793650794 | 91.95635301520403 | 91.93650793650792 | 3150 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 95.3015873015873 | 95.26786905757812 | 95.3015873015873 | 3150 | ok |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_right_pocket | 1 | 46.41269841269841 | 43.619322827044115 | 46.41269841269842 | 3150 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_right_pocket | 1 | 56.00000000000001 | 55.282438576375704 | 55.99999999999999 | 3150 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_right_pocket | 1 | 56.00000000000001 | 55.282438576375704 | 55.99999999999999 | 3150 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_right_pocket | 1 | 56.92063492063492 | 55.82783324373912 | 56.92063492063492 | 3150 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_right_pocket | 1 | 61.238095238095234 | 60.45761575700295 | 61.238095238095234 | 3150 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_right_pocket | 1 | 61.04761904761905 | 60.18483350424021 | 61.04761904761905 | 3150 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_right_pocket | 1 | 61.968253968253975 | 60.291627731701624 | 61.968253968253975 | 3150 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_right_pocket | 1 | 66.06349206349206 | 65.18557413669855 | 66.06349206349205 | 3150 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_right_pocket | 1 | 64.76190476190476 | 63.655380643537306 | 64.76190476190476 | 3150 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_right_pocket | 1 | 66.22222222222223 | 64.43760286844798 | 66.22222222222221 | 3150 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_right_pocket | 1 | 70.28571428571428 | 69.79285522976892 | 70.28571428571428 | 3150 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_right_pocket | 1 | 69.14285714285714 | 68.18220349165223 | 69.14285714285714 | 3150 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_right_pocket | 1 | 71.68253968253968 | 70.29123551293283 | 71.68253968253968 | 3150 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_right_pocket | 1 | 73.93650793650794 | 73.46743289554108 | 73.93650793650794 | 3150 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_right_pocket | 1 | 70.66666666666667 | 69.7863214744773 | 70.66666666666667 | 3150 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_right_pocket | 1 | 76.53968253968254 | 75.67100061418269 | 76.53968253968254 | 3150 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_right_pocket | 1 | 77.17460317460318 | 76.83318115766752 | 77.17460317460318 | 3150 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_right_pocket | 1 | 71.74603174603175 | 70.89147510365298 | 71.74603174603176 | 3150 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_right_pocket | 1 | 80.12698412698413 | 79.66755322230846 | 80.12698412698413 | 3150 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_right_pocket | 1 | 80.22222222222221 | 80.04082931849528 | 80.22222222222221 | 3150 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_right_pocket | 1 | 72.38095238095238 | 71.47519901679547 | 72.3809523809524 | 3150 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_right_pocket | 1 | 83.65079365079366 | 83.40105463290121 | 83.65079365079366 | 3150 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | phone_right_pocket | 1 | 82.25396825396825 | 82.0628037406822 | 82.25396825396825 | 3150 | ok |
| harnet | 4.49 | prototype | 128 | False | False | phone_right_pocket | 1 | 72.79365079365078 | 71.98183178592944 | 72.7936507936508 | 3150 | ok |
| harnet | 4.49 | ridge | 128 | False | False | phone_right_pocket | 1 | 86.12698412698413 | 85.98427571879007 | 86.12698412698413 | 3150 | ok |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_left_pocket | 1 | 48.857142857142854 | 45.883621869055204 | 48.85714285714286 | 3150 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_left_pocket | 1 | 55.01587301587302 | 54.097138443765424 | 55.015873015873005 | 3150 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_left_pocket | 1 | 55.01587301587302 | 54.097138443765424 | 55.015873015873005 | 3150 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_left_pocket | 1 | 55.9047619047619 | 55.00620942655951 | 55.9047619047619 | 3150 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_left_pocket | 1 | 61.904761904761905 | 61.317116507421154 | 61.9047619047619 | 3150 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_left_pocket | 1 | 62.63492063492063 | 61.85041735939614 | 62.63492063492063 | 3150 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_left_pocket | 1 | 63.20634920634921 | 62.0875955442974 | 63.2063492063492 | 3150 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_left_pocket | 1 | 66.85714285714286 | 66.14503728268234 | 66.85714285714285 | 3150 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_left_pocket | 1 | 67.11111111111111 | 66.38020883942232 | 67.11111111111111 | 3150 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_left_pocket | 1 | 68.57142857142857 | 67.45376731765803 | 68.57142857142857 | 3150 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_left_pocket | 1 | 71.23809523809524 | 70.83648406377712 | 71.23809523809524 | 3150 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_left_pocket | 1 | 70.76190476190476 | 70.1298876865063 | 70.76190476190477 | 3150 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_left_pocket | 1 | 73.61904761904762 | 72.86657343540006 | 73.61904761904762 | 3150 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_left_pocket | 1 | 76.06349206349206 | 75.79155510856495 | 76.06349206349206 | 3150 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_left_pocket | 1 | 73.36507936507937 | 72.5604378062565 | 73.36507936507938 | 3150 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_left_pocket | 1 | 78.0 | 77.44601398903328 | 78.00000000000001 | 3150 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_left_pocket | 1 | 78.60317460317461 | 78.35233541055489 | 78.6031746031746 | 3150 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_left_pocket | 1 | 74.15873015873015 | 73.63845246252015 | 74.15873015873015 | 3150 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_left_pocket | 1 | 80.31746031746032 | 79.97524018383999 | 80.31746031746032 | 3150 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_left_pocket | 1 | 81.30158730158729 | 81.1619412372744 | 81.3015873015873 | 3150 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_left_pocket | 1 | 74.31746031746032 | 73.77971113278903 | 74.31746031746033 | 3150 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_left_pocket | 1 | 82.95238095238095 | 82.67620728543096 | 82.95238095238095 | 3150 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | phone_left_pocket | 1 | 84.38095238095238 | 84.28264203959515 | 84.38095238095238 | 3150 | ok |
| harnet | 4.49 | prototype | 128 | False | False | phone_left_pocket | 1 | 74.98412698412699 | 74.46372447498601 | 74.98412698412696 | 3150 | ok |
| harnet | 4.49 | ridge | 128 | False | False | phone_left_pocket | 1 | 85.01587301587301 | 84.8241726761238 | 85.01587301587303 | 3150 | ok |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_belt | 1 | 49.111111111111114 | 47.411387553061054 | 49.111111111111114 | 3150 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_belt | 1 | 52.539682539682545 | 51.576493127731204 | 52.539682539682545 | 3150 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_belt | 1 | 52.539682539682545 | 51.576493127731204 | 52.539682539682545 | 3150 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_belt | 1 | 52.03174603174603 | 50.705494343344014 | 52.03174603174603 | 3150 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_belt | 1 | 59.36507936507937 | 58.49774586195888 | 59.36507936507935 | 3150 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_belt | 1 | 59.26984126984127 | 58.57514095854113 | 59.26984126984127 | 3150 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_belt | 1 | 58.857142857142854 | 57.113177737913766 | 58.85714285714286 | 3150 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_belt | 1 | 62.82539682539683 | 62.21757915518479 | 62.82539682539683 | 3150 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_belt | 1 | 64.22222222222223 | 63.57609309487518 | 64.22222222222223 | 3150 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_belt | 1 | 65.07936507936508 | 63.76896268228295 | 65.07936507936508 | 3150 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_belt | 1 | 67.4920634920635 | 66.91917039354722 | 67.49206349206351 | 3150 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_belt | 1 | 68.63492063492063 | 68.15143949026312 | 68.63492063492063 | 3150 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_belt | 1 | 70.66666666666667 | 69.76719913470582 | 70.66666666666667 | 3150 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_belt | 1 | 71.33333333333334 | 70.86568880406648 | 71.33333333333333 | 3150 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_belt | 1 | 72.15873015873015 | 71.79324962653514 | 72.15873015873015 | 3150 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_belt | 1 | 75.61904761904762 | 75.0317809485713 | 75.61904761904762 | 3150 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_belt | 1 | 74.03174603174602 | 73.66794629442464 | 74.03174603174604 | 3150 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_belt | 1 | 73.23809523809524 | 72.92961882707318 | 73.23809523809524 | 3150 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_belt | 1 | 80.19047619047619 | 79.88878911777552 | 80.1904761904762 | 3150 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_belt | 1 | 76.7936507936508 | 76.50272752081221 | 76.7936507936508 | 3150 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_belt | 1 | 74.85714285714286 | 74.57220323056085 | 74.85714285714286 | 3150 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_belt | 1 | 83.68253968253968 | 83.48465965087898 | 83.68253968253968 | 3150 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | phone_belt | 1 | 78.85714285714286 | 78.64787059191588 | 78.85714285714285 | 3150 | ok |
| harnet | 4.49 | prototype | 128 | False | False | phone_belt | 1 | 76.06349206349206 | 75.84872139474797 | 76.06349206349206 | 3150 | ok |
| harnet | 4.49 | ridge | 128 | False | False | phone_belt | 1 | 86.19047619047619 | 86.06039850022064 | 86.19047619047619 | 3150 | ok |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | watch_wrist_proxy | 1 | 57.65079365079365 | 55.12665888366155 | 57.65079365079365 | 3150 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | watch_wrist_proxy | 1 | 57.26984126984127 | 56.370243366464024 | 57.26984126984126 | 3150 | ok |
| harnet | 4.49 | prototype | 1 | False | False | watch_wrist_proxy | 1 | 57.26984126984127 | 56.370243366464024 | 57.26984126984126 | 3150 | ok |
| harnet | 4.49 | ridge | 1 | False | False | watch_wrist_proxy | 1 | 57.079365079365076 | 55.693925359963025 | 57.079365079365076 | 3150 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | watch_wrist_proxy | 1 | 64.34920634920634 | 63.76042417884189 | 64.34920634920634 | 3150 | ok |
| harnet | 4.49 | prototype | 2 | False | False | watch_wrist_proxy | 1 | 63.07936507936508 | 62.47849180246716 | 63.07936507936508 | 3150 | ok |
| harnet | 4.49 | ridge | 2 | False | False | watch_wrist_proxy | 1 | 62.73015873015873 | 61.38904256837089 | 62.730158730158735 | 3150 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | watch_wrist_proxy | 1 | 66.88888888888889 | 66.54007604157063 | 66.88888888888889 | 3150 | ok |
| harnet | 4.49 | prototype | 4 | False | False | watch_wrist_proxy | 1 | 66.98412698412697 | 66.4047196996474 | 66.98412698412699 | 3150 | ok |
| harnet | 4.49 | ridge | 4 | False | False | watch_wrist_proxy | 1 | 66.60317460317461 | 65.3813361658921 | 66.60317460317461 | 3150 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | watch_wrist_proxy | 1 | 69.14285714285714 | 68.69887008689011 | 69.14285714285714 | 3150 | ok |
| harnet | 4.49 | prototype | 8 | False | False | watch_wrist_proxy | 1 | 68.73015873015873 | 68.35474047825097 | 68.73015873015872 | 3150 | ok |
| harnet | 4.49 | ridge | 8 | False | False | watch_wrist_proxy | 1 | 69.9047619047619 | 68.81292967310621 | 69.9047619047619 | 3150 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | watch_wrist_proxy | 1 | 71.61904761904762 | 71.32059134165331 | 71.61904761904762 | 3150 | ok |
| harnet | 4.49 | prototype | 16 | False | False | watch_wrist_proxy | 1 | 71.42857142857143 | 71.20076445096643 | 71.42857142857144 | 3150 | ok |
| harnet | 4.49 | ridge | 16 | False | False | watch_wrist_proxy | 1 | 74.95238095238095 | 74.31589174352263 | 74.95238095238095 | 3150 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | watch_wrist_proxy | 1 | 71.26984126984127 | 71.02746204600045 | 71.26984126984127 | 3150 | ok |
| harnet | 4.49 | prototype | 32 | False | False | watch_wrist_proxy | 1 | 74.31746031746032 | 74.18678044608045 | 74.31746031746032 | 3150 | ok |
| harnet | 4.49 | ridge | 32 | False | False | watch_wrist_proxy | 1 | 78.82539682539682 | 78.36474762485594 | 78.82539682539681 | 3150 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | watch_wrist_proxy | 1 | 74.85714285714286 | 74.7427690967289 | 74.85714285714286 | 3150 | ok |
| harnet | 4.49 | prototype | 64 | False | False | watch_wrist_proxy | 1 | 75.3015873015873 | 75.18631808615301 | 75.3015873015873 | 3150 | ok |
| harnet | 4.49 | ridge | 64 | False | False | watch_wrist_proxy | 1 | 82.15873015873015 | 81.90639388147943 | 82.15873015873017 | 3150 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | watch_wrist_proxy | 1 | 77.23809523809524 | 77.13111831068264 | 77.23809523809526 | 3150 | ok |
| harnet | 4.49 | prototype | 128 | False | False | watch_wrist_proxy | 1 | 75.87301587301587 | 75.77948332615094 | 75.87301587301587 | 3150 | ok |
| harnet | 4.49 | ridge | 128 | False | False | watch_wrist_proxy | 1 | 83.4920634920635 | 83.36755324065798 | 83.4920634920635 | 3150 | ok |
| harnet | 4.49 | zero_support | 0 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| harnet | 4.49 | 1nn | 1 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 64.5079365079365 | 63.79831565376851 | 64.50793650793652 | 3150 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 64.5079365079365 | 63.79831565376851 | 64.50793650793652 | 3150 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 63.84126984126984 | 62.527961551485056 | 63.841269841269835 | 3150 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 68.06349206349206 | 67.23132954271772 | 68.06349206349208 | 3150 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 67.93650793650794 | 67.06204906406612 | 67.93650793650794 | 3150 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 66.19047619047619 | 64.65309691083456 | 66.19047619047619 | 3150 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 73.33333333333333 | 72.8188019853389 | 73.33333333333334 | 3150 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 71.84126984126983 | 71.174326403319 | 71.84126984126983 | 3150 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 70.73015873015873 | 69.63438865232919 | 70.73015873015872 | 3150 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 76.92063492063492 | 76.58567709061698 | 76.92063492063491 | 3150 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 74.85714285714286 | 74.25558109236395 | 74.85714285714286 | 3150 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 75.04761904761905 | 74.32428450186714 | 75.04761904761904 | 3150 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 79.55555555555556 | 79.25587452883815 | 79.55555555555556 | 3150 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 76.88888888888889 | 76.35239350099415 | 76.8888888888889 | 3150 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 79.4920634920635 | 78.93307643216615 | 79.4920634920635 | 3150 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 83.52380952380952 | 83.38049065199219 | 83.52380952380952 | 3150 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 78.41269841269842 | 77.87491781777618 | 78.41269841269842 | 3150 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 82.22222222222221 | 81.83690004730816 | 82.22222222222221 | 3150 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 85.39682539682539 | 85.31100479637507 | 85.39682539682539 | 3150 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 78.63492063492063 | 78.12735282992026 | 78.63492063492063 | 3150 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 83.9047619047619 | 83.55247881736456 | 83.9047619047619 | 3150 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 86.6984126984127 | 86.61016981006317 | 86.6984126984127 | 3150 | ok |
| harnet | 4.49 | prototype | 128 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 79.23809523809524 | 78.75641841602317 | 79.23809523809524 | 3150 | ok |
| harnet | 4.49 | ridge | 128 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 85.74603174603175 | 85.48462091798409 | 85.74603174603175 | 3150 | ok |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_right_pocket | 1 | 40.79365079365079 | 33.66968751529337 | 40.79365079365079 | 3150 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_right_pocket | 1 | 75.96825396825398 | 75.87855338700727 | 75.96825396825396 | 3150 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_right_pocket | 1 | 75.96825396825398 | 75.87855338700727 | 75.96825396825396 | 3150 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_right_pocket | 1 | 75.04761904761905 | 74.40842640020662 | 75.04761904761904 | 3150 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_right_pocket | 1 | 82.95238095238095 | 82.92153438112479 | 82.95238095238095 | 3150 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_right_pocket | 1 | 81.9047619047619 | 81.86946279988062 | 81.9047619047619 | 3150 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_right_pocket | 1 | 81.4920634920635 | 80.98406144148173 | 81.4920634920635 | 3150 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_right_pocket | 1 | 89.33333333333333 | 89.34931397901119 | 89.33333333333334 | 3150 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_right_pocket | 1 | 85.2063492063492 | 85.04065859673105 | 85.2063492063492 | 3150 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_right_pocket | 1 | 84.92063492063492 | 84.49462860369496 | 84.92063492063492 | 3150 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_right_pocket | 1 | 91.3015873015873 | 91.33751205022672 | 91.30158730158729 | 3150 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_right_pocket | 1 | 86.47619047619047 | 86.15363467342281 | 86.47619047619048 | 3150 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_right_pocket | 1 | 86.09523809523809 | 85.55842349672005 | 86.09523809523809 | 3150 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_right_pocket | 1 | 93.87301587301587 | 93.88526534712544 | 93.87301587301586 | 3150 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_right_pocket | 1 | 87.87301587301587 | 87.52206798384192 | 87.87301587301587 | 3150 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_right_pocket | 1 | 87.39682539682539 | 86.62877996131797 | 87.39682539682539 | 3150 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_right_pocket | 1 | 94.85714285714286 | 94.87514106027311 | 94.85714285714285 | 3150 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_right_pocket | 1 | 88.19047619047619 | 87.79040348055722 | 88.1904761904762 | 3150 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_right_pocket | 1 | 88.53968253968254 | 87.86703373250691 | 88.53968253968254 | 3150 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_right_pocket | 1 | 96.03174603174604 | 96.03807655384465 | 96.03174603174605 | 3150 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_right_pocket | 1 | 89.07936507936508 | 88.75105295753448 | 89.07936507936508 | 3150 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_right_pocket | 1 | 91.87301587301587 | 91.66130875640744 | 91.87301587301587 | 3150 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | phone_right_pocket | 1 | 96.5079365079365 | 96.51378959478602 | 96.50793650793652 | 3150 | ok |
| unimts | 68.61 | prototype | 128 | True | False | phone_right_pocket | 1 | 89.33333333333333 | 89.01997309253883 | 89.33333333333334 | 3150 | ok |
| unimts | 68.61 | ridge | 128 | True | False | phone_right_pocket | 1 | 95.11111111111111 | 95.10029769345148 | 95.11111111111111 | 3150 | ok |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_left_pocket | 1 | 40.19047619047619 | 33.04895167488443 | 40.19047619047619 | 3150 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_left_pocket | 1 | 76.82539682539684 | 76.68919995029457 | 76.82539682539684 | 3150 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_left_pocket | 1 | 76.82539682539684 | 76.68919995029457 | 76.82539682539684 | 3150 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_left_pocket | 1 | 77.68253968253968 | 77.04274903736035 | 77.68253968253967 | 3150 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_left_pocket | 1 | 85.33333333333334 | 85.19472809701337 | 85.33333333333334 | 3150 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_left_pocket | 1 | 83.33333333333334 | 83.21518946756976 | 83.33333333333333 | 3150 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_left_pocket | 1 | 83.42857142857143 | 82.94235493448564 | 83.42857142857144 | 3150 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_left_pocket | 1 | 90.47619047619048 | 90.48441822739004 | 90.47619047619048 | 3150 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_left_pocket | 1 | 86.6984126984127 | 86.53771630751675 | 86.6984126984127 | 3150 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_left_pocket | 1 | 86.15873015873015 | 85.77925231926027 | 86.15873015873015 | 3150 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_left_pocket | 1 | 91.55555555555556 | 91.53375260516141 | 91.55555555555556 | 3150 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_left_pocket | 1 | 88.38095238095238 | 88.2252090781231 | 88.38095238095238 | 3150 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_left_pocket | 1 | 87.96825396825398 | 87.55008890022168 | 87.96825396825396 | 3150 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_left_pocket | 1 | 93.71428571428572 | 93.71277900244817 | 93.71428571428571 | 3150 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_left_pocket | 1 | 90.88888888888889 | 90.75884564296874 | 90.8888888888889 | 3150 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_left_pocket | 1 | 90.28571428571428 | 89.99320052869699 | 90.28571428571429 | 3150 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_left_pocket | 1 | 94.73015873015873 | 94.72790505989738 | 94.73015873015873 | 3150 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_left_pocket | 1 | 92.6984126984127 | 92.64193324052967 | 92.6984126984127 | 3150 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_left_pocket | 1 | 92.98412698412697 | 92.88769117155168 | 92.98412698412697 | 3150 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_left_pocket | 1 | 95.36507936507937 | 95.36200713790988 | 95.36507936507937 | 3150 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_left_pocket | 1 | 93.65079365079364 | 93.62081229716136 | 93.65079365079364 | 3150 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_left_pocket | 1 | 95.01587301587303 | 94.99791941540884 | 95.01587301587301 | 3150 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | phone_left_pocket | 1 | 95.93650793650794 | 95.93326687905976 | 95.93650793650794 | 3150 | ok |
| unimts | 68.61 | prototype | 128 | True | False | phone_left_pocket | 1 | 94.47619047619048 | 94.46175887475276 | 94.47619047619048 | 3150 | ok |
| unimts | 68.61 | ridge | 128 | True | False | phone_left_pocket | 1 | 96.06349206349206 | 96.05988790418853 | 96.06349206349208 | 3150 | ok |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_belt | 1 | 53.84126984126985 | 45.06513031728288 | 53.841269841269835 | 3150 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_belt | 1 | 68.76190476190476 | 68.00386412690732 | 68.76190476190477 | 3150 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_belt | 1 | 68.76190476190476 | 68.00386412690732 | 68.76190476190477 | 3150 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_belt | 1 | 64.66666666666666 | 63.15904562485202 | 64.66666666666666 | 3150 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_belt | 1 | 75.65079365079364 | 75.14271838936396 | 75.65079365079367 | 3150 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_belt | 1 | 75.33333333333333 | 74.89171619968396 | 75.33333333333333 | 3150 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_belt | 1 | 70.98412698412699 | 69.78237219009392 | 70.98412698412699 | 3150 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_belt | 1 | 79.04761904761905 | 78.79283479989827 | 79.04761904761905 | 3150 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_belt | 1 | 78.28571428571428 | 78.06732112711096 | 78.28571428571428 | 3150 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_belt | 1 | 76.0 | 75.26865753418711 | 76.0 | 3150 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_belt | 1 | 82.28571428571428 | 82.07106768877324 | 82.28571428571428 | 3150 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_belt | 1 | 81.52380952380952 | 81.38361197890632 | 81.52380952380953 | 3150 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_belt | 1 | 78.98412698412697 | 78.45824585443849 | 78.98412698412699 | 3150 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_belt | 1 | 84.38095238095238 | 84.17020870388441 | 84.38095238095238 | 3150 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_belt | 1 | 82.95238095238095 | 82.76695356972358 | 82.95238095238095 | 3150 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_belt | 1 | 83.01587301587303 | 82.65006511759282 | 83.01587301587303 | 3150 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_belt | 1 | 85.74603174603175 | 85.58386933952833 | 85.74603174603173 | 3150 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_belt | 1 | 84.57142857142857 | 84.44772939218635 | 84.57142857142857 | 3150 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_belt | 1 | 86.25396825396825 | 86.0717180826909 | 86.25396825396825 | 3150 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_belt | 1 | 87.46031746031746 | 87.34966041030118 | 87.46031746031747 | 3150 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_belt | 1 | 85.17460317460316 | 85.07064762276049 | 85.17460317460316 | 3150 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_belt | 1 | 87.77777777777777 | 87.64281118132874 | 87.77777777777779 | 3150 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | phone_belt | 1 | 88.34920634920634 | 88.25616426729168 | 88.34920634920634 | 3150 | ok |
| unimts | 68.61 | prototype | 128 | True | False | phone_belt | 1 | 85.11111111111111 | 85.02213078501596 | 85.11111111111111 | 3150 | ok |
| unimts | 68.61 | ridge | 128 | True | False | phone_belt | 1 | 89.23809523809524 | 89.15365897159815 | 89.23809523809524 | 3150 | ok |
| unimts | 68.61 | native_zero_support | 0 | True | False | watch_wrist_proxy | 1 | 45.52380952380952 | 35.49308798812523 | 45.52380952380952 | 3150 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | watch_wrist_proxy | 1 | 65.58730158730158 | 65.67771373644042 | 65.58730158730158 | 3150 | ok |
| unimts | 68.61 | prototype | 1 | True | False | watch_wrist_proxy | 1 | 65.58730158730158 | 65.67771373644042 | 65.58730158730158 | 3150 | ok |
| unimts | 68.61 | ridge | 1 | True | False | watch_wrist_proxy | 1 | 58.730158730158735 | 57.722898723419114 | 58.73015873015872 | 3150 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | watch_wrist_proxy | 1 | 73.4920634920635 | 73.52857754158734 | 73.49206349206348 | 3150 | ok |
| unimts | 68.61 | prototype | 2 | True | False | watch_wrist_proxy | 1 | 69.07936507936509 | 69.28656007882141 | 69.07936507936509 | 3150 | ok |
| unimts | 68.61 | ridge | 2 | True | False | watch_wrist_proxy | 1 | 62.50793650793651 | 61.866716976348194 | 62.50793650793651 | 3150 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | watch_wrist_proxy | 1 | 76.57142857142857 | 76.53790867569056 | 76.57142857142858 | 3150 | ok |
| unimts | 68.61 | prototype | 4 | True | False | watch_wrist_proxy | 1 | 70.5079365079365 | 70.70454253971616 | 70.5079365079365 | 3150 | ok |
| unimts | 68.61 | ridge | 4 | True | False | watch_wrist_proxy | 1 | 66.0952380952381 | 65.935078481998 | 66.0952380952381 | 3150 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | watch_wrist_proxy | 1 | 79.36507936507937 | 79.44466661267765 | 79.36507936507937 | 3150 | ok |
| unimts | 68.61 | prototype | 8 | True | False | watch_wrist_proxy | 1 | 71.36507936507937 | 71.5300243998399 | 71.36507936507937 | 3150 | ok |
| unimts | 68.61 | ridge | 8 | True | False | watch_wrist_proxy | 1 | 69.4920634920635 | 69.2710950091329 | 69.4920634920635 | 3150 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | watch_wrist_proxy | 1 | 80.82539682539682 | 80.83973023110497 | 80.82539682539684 | 3150 | ok |
| unimts | 68.61 | prototype | 16 | True | False | watch_wrist_proxy | 1 | 72.09523809523809 | 72.17230182305026 | 72.09523809523809 | 3150 | ok |
| unimts | 68.61 | ridge | 16 | True | False | watch_wrist_proxy | 1 | 72.06349206349206 | 71.81139895001573 | 72.06349206349205 | 3150 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | watch_wrist_proxy | 1 | 82.5079365079365 | 82.62736010687901 | 82.50793650793649 | 3150 | ok |
| unimts | 68.61 | prototype | 32 | True | False | watch_wrist_proxy | 1 | 73.80952380952381 | 73.83787693639736 | 73.80952380952381 | 3150 | ok |
| unimts | 68.61 | ridge | 32 | True | False | watch_wrist_proxy | 1 | 74.98412698412699 | 74.69282653640784 | 74.98412698412699 | 3150 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | watch_wrist_proxy | 1 | 84.06349206349206 | 84.1558916946058 | 84.06349206349206 | 3150 | ok |
| unimts | 68.61 | prototype | 64 | True | False | watch_wrist_proxy | 1 | 77.11111111111111 | 77.10511370502059 | 77.11111111111111 | 3150 | ok |
| unimts | 68.61 | ridge | 64 | True | False | watch_wrist_proxy | 1 | 82.19047619047619 | 82.06169702059893 | 82.19047619047618 | 3150 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | watch_wrist_proxy | 1 | 84.88888888888889 | 85.02655847021984 | 84.88888888888889 | 3150 | ok |
| unimts | 68.61 | prototype | 128 | True | False | watch_wrist_proxy | 1 | 79.61904761904762 | 79.60101435101863 | 79.61904761904762 | 3150 | ok |
| unimts | 68.61 | ridge | 128 | True | False | watch_wrist_proxy | 1 | 86.73015873015873 | 86.69964053579126 | 86.73015873015872 | 3150 | ok |
| unimts | 68.61 | zero_support | 0 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| unimts | 68.61 | 1nn | 1 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 72.88888888888889 | 72.84657955267163 | 72.88888888888889 | 3150 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 72.88888888888889 | 72.84657955267163 | 72.88888888888889 | 3150 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 73.87301587301587 | 73.47116632673897 | 73.87301587301587 | 3150 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 82.25396825396825 | 82.26592288363929 | 82.25396825396825 | 3150 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 79.04761904761905 | 79.05094139134968 | 79.04761904761905 | 3150 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 79.14285714285715 | 78.78645975800787 | 79.14285714285715 | 3150 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 86.76190476190476 | 86.75059779273944 | 86.76190476190477 | 3150 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 81.68253968253968 | 81.64026892606479 | 81.68253968253968 | 3150 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 81.14285714285714 | 80.89779154365816 | 81.14285714285714 | 3150 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 90.15873015873017 | 90.13098950780528 | 90.15873015873017 | 3150 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 84.92063492063492 | 84.79162090562934 | 84.92063492063492 | 3150 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 85.36507936507937 | 85.16609936177588 | 85.36507936507937 | 3150 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 92.41269841269842 | 92.41306523379464 | 92.41269841269842 | 3150 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 86.12698412698413 | 85.90060163080061 | 86.12698412698413 | 3150 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 88.5079365079365 | 88.19704679082673 | 88.5079365079365 | 3150 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 93.42857142857143 | 93.45288926441275 | 93.42857142857143 | 3150 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 87.33333333333333 | 87.11353386654405 | 87.33333333333333 | 3150 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 91.68253968253968 | 91.57483763892479 | 91.68253968253968 | 3150 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 94.25396825396825 | 94.2618359437879 | 94.25396825396827 | 3150 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 87.17460317460318 | 86.88375315122869 | 87.17460317460318 | 3150 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 93.9047619047619 | 93.89245847144225 | 93.9047619047619 | 3150 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 95.2063492063492 | 95.21424070735328 | 95.2063492063492 | 3150 | ok |
| unimts | 68.61 | prototype | 128 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 87.36507936507937 | 87.0968754805674 | 87.36507936507935 | 3150 | ok |
| unimts | 68.61 | ridge | 128 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 95.4920634920635 | 95.49679705815322 | 95.49206349206348 | 3150 | ok |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_right_pocket | 1 | 14.285714285714285 | 3.571428571428571 | 14.285714285714285 | 3150 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_right_pocket | 1 | 21.206349206349206 | 21.227456075091883 | 21.206349206349202 | 3150 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_right_pocket | 1 | 21.206349206349206 | 21.227456075091883 | 21.206349206349202 | 3150 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_right_pocket | 1 | 20.793650793650794 | 20.477660733142056 | 20.79365079365079 | 3150 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_right_pocket | 1 | 23.682539682539684 | 23.80927625871829 | 23.682539682539687 | 3150 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_right_pocket | 1 | 21.746031746031747 | 21.72711870932976 | 21.74603174603174 | 3150 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_right_pocket | 1 | 21.206349206349206 | 20.763586797380924 | 21.206349206349206 | 3150 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_right_pocket | 1 | 27.269841269841272 | 27.581358147571304 | 27.269841269841265 | 3150 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_right_pocket | 1 | 24.03174603174603 | 23.73411458660304 | 24.031746031746028 | 3150 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_right_pocket | 1 | 23.523809523809526 | 22.512763813036223 | 23.523809523809522 | 3150 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_right_pocket | 1 | 30.793650793650794 | 31.043181721848562 | 30.79365079365079 | 3150 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_right_pocket | 1 | 24.095238095238095 | 23.59515817411294 | 24.0952380952381 | 3150 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_right_pocket | 1 | 23.809523809523807 | 22.22208944617688 | 23.809523809523807 | 3150 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_right_pocket | 1 | 36.666666666666664 | 37.406507743267916 | 36.666666666666664 | 3150 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_right_pocket | 1 | 24.698412698412696 | 23.715346022892845 | 24.698412698412696 | 3150 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_right_pocket | 1 | 24.952380952380953 | 21.974559448956775 | 24.952380952380953 | 3150 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_right_pocket | 1 | 40.476190476190474 | 41.06396208372294 | 40.476190476190474 | 3150 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_right_pocket | 1 | 25.428571428571427 | 23.670706163128838 | 25.428571428571427 | 3150 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_right_pocket | 1 | 27.015873015873016 | 23.194963838955562 | 27.015873015873016 | 3150 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_right_pocket | 1 | 44.25396825396825 | 44.892481366096895 | 44.25396825396825 | 3150 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_right_pocket | 1 | 24.793650793650794 | 22.066418487054865 | 24.793650793650794 | 3150 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_right_pocket | 1 | 30.28571428571429 | 26.35305688483866 | 30.28571428571428 | 3150 | ok |
| normwear | 1293.86 | 1nn | 128 | True | False | phone_right_pocket | 1 | 47.80952380952381 | 48.58628618320723 | 47.80952380952381 | 3150 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | phone_right_pocket | 1 | 25.9047619047619 | 22.226247622529712 | 25.9047619047619 | 3150 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | phone_right_pocket | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_left_pocket | 1 | 14.285714285714285 | 3.571428571428571 | 14.285714285714285 | 3150 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_left_pocket | 1 | 22.92063492063492 | 23.028965224978634 | 22.92063492063492 | 3150 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_left_pocket | 1 | 22.92063492063492 | 23.028965224978634 | 22.92063492063492 | 3150 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_left_pocket | 1 | 20.793650793650794 | 20.278702942925015 | 20.793650793650794 | 3150 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_left_pocket | 1 | 23.841269841269842 | 24.03638544508393 | 23.841269841269842 | 3150 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_left_pocket | 1 | 22.158730158730158 | 22.110428845767622 | 22.158730158730155 | 3150 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_left_pocket | 1 | 21.555555555555557 | 20.913303059040466 | 21.555555555555557 | 3150 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_left_pocket | 1 | 26.190476190476193 | 26.58000127324538 | 26.190476190476186 | 3150 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_left_pocket | 1 | 23.42857142857143 | 23.047720696798805 | 23.42857142857143 | 3150 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_left_pocket | 1 | 23.96825396825397 | 22.26658663901707 | 23.96825396825397 | 3150 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_left_pocket | 1 | 31.873015873015873 | 32.40864668223831 | 31.87301587301587 | 3150 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_left_pocket | 1 | 25.11111111111111 | 24.335783168192513 | 25.11111111111111 | 3150 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_left_pocket | 1 | 23.555555555555554 | 20.770532679685232 | 23.555555555555554 | 3150 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_left_pocket | 1 | 35.20634920634921 | 35.89963729360729 | 35.2063492063492 | 3150 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_left_pocket | 1 | 25.26984126984127 | 23.721146571559654 | 25.26984126984127 | 3150 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_left_pocket | 1 | 25.333333333333336 | 21.21900486633126 | 25.333333333333336 | 3150 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_left_pocket | 1 | 41.269841269841265 | 41.78823896835749 | 41.26984126984127 | 3150 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_left_pocket | 1 | 25.936507936507937 | 23.389927948579377 | 25.936507936507937 | 3150 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_left_pocket | 1 | 26.222222222222225 | 21.3065573075995 | 26.222222222222218 | 3150 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_left_pocket | 1 | 43.65079365079365 | 44.17373730614306 | 43.65079365079366 | 3150 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_left_pocket | 1 | 26.79365079365079 | 23.09604573931988 | 26.793650793650798 | 3150 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_left_pocket | 1 | 30.571428571428573 | 25.822163953872263 | 30.571428571428566 | 3150 | ok |
| normwear | 1293.86 | 1nn | 128 | True | False | phone_left_pocket | 1 | 47.269841269841265 | 47.6949850486078 | 47.269841269841265 | 3150 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | phone_left_pocket | 1 | 26.6984126984127 | 22.582322389570038 | 26.6984126984127 | 3150 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | phone_left_pocket | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_belt | 1 | 14.285714285714285 | 3.571428571428571 | 14.285714285714285 | 3150 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_belt | 1 | 24.285714285714285 | 24.357586899661726 | 24.285714285714285 | 3150 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_belt | 1 | 24.285714285714285 | 24.357586899661726 | 24.285714285714285 | 3150 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_belt | 1 | 24.349206349206348 | 23.400501772160013 | 24.349206349206348 | 3150 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_belt | 1 | 26.6984126984127 | 26.793810557169266 | 26.6984126984127 | 3150 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_belt | 1 | 27.523809523809522 | 27.307847702574424 | 27.523809523809522 | 3150 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_belt | 1 | 25.142857142857146 | 23.763953520026277 | 25.14285714285714 | 3150 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_belt | 1 | 28.6984126984127 | 28.754981793289897 | 28.6984126984127 | 3150 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_belt | 1 | 27.904761904761905 | 27.64988188182696 | 27.90476190476191 | 3150 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_belt | 1 | 25.873015873015877 | 23.77331504173962 | 25.873015873015877 | 3150 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_belt | 1 | 31.77777777777778 | 31.84457576999985 | 31.77777777777778 | 3150 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_belt | 1 | 27.873015873015873 | 27.46546726865885 | 27.873015873015873 | 3150 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_belt | 1 | 26.53968253968254 | 23.638977641073318 | 26.53968253968254 | 3150 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_belt | 1 | 35.301587301587304 | 35.39889731962466 | 35.301587301587304 | 3150 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_belt | 1 | 29.873015873015873 | 29.56042749834042 | 29.873015873015873 | 3150 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_belt | 1 | 28.253968253968253 | 25.036177032292123 | 28.253968253968253 | 3150 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_belt | 1 | 38.57142857142858 | 38.754776402698006 | 38.57142857142857 | 3150 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_belt | 1 | 29.619047619047617 | 29.25635290389072 | 29.619047619047617 | 3150 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_belt | 1 | 30.507936507936506 | 26.878230106696044 | 30.507936507936506 | 3150 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_belt | 1 | 41.58730158730159 | 41.71135609940425 | 41.58730158730159 | 3150 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_belt | 1 | 29.23809523809524 | 28.505864519589903 | 29.238095238095234 | 3150 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_belt | 1 | 31.77777777777778 | 27.431546222340792 | 31.77777777777778 | 3150 | ok |
| normwear | 1293.86 | 1nn | 128 | True | False | phone_belt | 1 | 46.15873015873016 | 46.36180577896677 | 46.158730158730165 | 3150 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | phone_belt | 1 | 28.063492063492063 | 27.07396604299128 | 28.063492063492063 | 3150 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | phone_belt | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | watch_wrist_proxy | 1 | 14.285714285714285 | 3.571428571428571 | 14.285714285714285 | 3150 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | watch_wrist_proxy | 1 | 26.031746031746035 | 25.95475830680473 | 26.031746031746035 | 3150 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | watch_wrist_proxy | 1 | 26.031746031746035 | 25.95475830680473 | 26.031746031746035 | 3150 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | watch_wrist_proxy | 1 | 24.793650793650794 | 23.856688390731552 | 24.793650793650794 | 3150 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | watch_wrist_proxy | 1 | 31.841269841269842 | 31.88794837574685 | 31.841269841269842 | 3150 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | watch_wrist_proxy | 1 | 27.873015873015873 | 27.548099527683107 | 27.873015873015873 | 3150 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | watch_wrist_proxy | 1 | 25.07936507936508 | 23.388182870736827 | 25.07936507936508 | 3150 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | watch_wrist_proxy | 1 | 38.19047619047619 | 38.36309312993614 | 38.19047619047619 | 3150 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | watch_wrist_proxy | 1 | 31.396825396825395 | 30.44389781098893 | 31.396825396825395 | 3150 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | watch_wrist_proxy | 1 | 29.682539682539684 | 25.862128276628088 | 29.682539682539677 | 3150 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | watch_wrist_proxy | 1 | 40.98412698412698 | 40.92739018182635 | 40.98412698412699 | 3150 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | watch_wrist_proxy | 1 | 32.666666666666664 | 30.079052301754235 | 32.666666666666664 | 3150 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | watch_wrist_proxy | 1 | 31.80952380952381 | 25.173108875197787 | 31.80952380952381 | 3150 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | watch_wrist_proxy | 1 | 47.01587301587302 | 47.070073936334396 | 47.01587301587301 | 3150 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | watch_wrist_proxy | 1 | 32.79365079365079 | 29.492165328336235 | 32.79365079365079 | 3150 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | watch_wrist_proxy | 1 | 34.76190476190476 | 26.65863587959792 | 34.761904761904766 | 3150 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | watch_wrist_proxy | 1 | 50.41269841269841 | 50.413322108859305 | 50.41269841269841 | 3150 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | watch_wrist_proxy | 1 | 35.079365079365076 | 30.885672081850323 | 35.079365079365076 | 3150 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | watch_wrist_proxy | 1 | 37.87301587301587 | 28.172897291210713 | 37.87301587301587 | 3150 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | watch_wrist_proxy | 1 | 54.158730158730165 | 54.28186699590506 | 54.158730158730165 | 3150 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | watch_wrist_proxy | 1 | 34.41269841269841 | 29.62068105207155 | 34.41269841269841 | 3150 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | watch_wrist_proxy | 1 | 38.38095238095238 | 28.27545845302259 | 38.38095238095238 | 3150 | ok |
| normwear | 1293.86 | 1nn | 128 | True | False | watch_wrist_proxy | 1 | 58.88888888888889 | 58.99998751271136 | 58.88888888888888 | 3150 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | watch_wrist_proxy | 1 | 34.82539682539682 | 29.949240301645485 | 34.82539682539682 | 3150 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | watch_wrist_proxy | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | zero_support | 0 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 25.746031746031743 | 25.846460689666063 | 25.746031746031743 | 3150 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 25.746031746031743 | 25.846460689666063 | 25.746031746031743 | 3150 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 23.936507936507937 | 23.299200459932944 | 23.93650793650794 | 3150 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 29.3968253968254 | 29.45966942413903 | 29.3968253968254 | 3150 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 29.619047619047617 | 29.6090287758875 | 29.619047619047624 | 3150 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 26.095238095238095 | 24.97565638001284 | 26.095238095238095 | 3150 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 33.74603174603175 | 33.869971983438944 | 33.74603174603174 | 3150 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 29.11111111111111 | 28.744553524385392 | 29.111111111111104 | 3150 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 25.11111111111111 | 22.383588059006314 | 25.11111111111111 | 3150 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 39.77777777777778 | 39.83917852575554 | 39.77777777777778 | 3150 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 31.587301587301585 | 30.91615158133902 | 31.587301587301592 | 3150 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 27.111111111111114 | 23.319607105059156 | 27.111111111111114 | 3150 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 45.111111111111114 | 45.159108695853334 | 45.11111111111111 | 3150 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 32.666666666666664 | 31.505091437385364 | 32.666666666666664 | 3150 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 28.6984126984127 | 23.279306300939833 | 28.6984126984127 | 3150 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 50.79365079365079 | 50.91922013521016 | 50.793650793650805 | 3150 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 33.523809523809526 | 31.936008234189444 | 33.523809523809526 | 3150 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 32.88888888888889 | 27.12519921524433 | 32.888888888888886 | 3150 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 56.41269841269841 | 56.60800771329342 | 56.41269841269841 | 3150 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 33.682539682539684 | 31.745833755984382 | 33.68253968253968 | 3150 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 37.111111111111114 | 32.08439512666394 | 37.11111111111111 | 3150 | ok |
| normwear | 1293.86 | 1nn | 128 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 58.76190476190476 | 58.956245631641636 | 58.76190476190476 | 3150 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 32.44444444444444 | 30.49831151714036 | 32.44444444444445 | 3150 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_right_pocket | 1 | 43.17460317460318 | 35.756496482338285 | 43.17460317460317 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_right_pocket | 1 | 67.26984126984127 | 67.64002106907922 | 67.26984126984127 | 3150 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_right_pocket | 1 | 67.26984126984127 | 67.64002106907922 | 67.26984126984127 | 3150 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_right_pocket | 1 | 61.968253968253975 | 58.89880412628441 | 61.968253968253975 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_right_pocket | 1 | 75.4920634920635 | 75.63790350819568 | 75.4920634920635 | 3150 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_right_pocket | 1 | 75.55555555555556 | 75.74446397905979 | 75.55555555555557 | 3150 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_right_pocket | 1 | 64.5079365079365 | 61.592501927032174 | 64.5079365079365 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_right_pocket | 1 | 82.25396825396825 | 82.32850811675318 | 82.25396825396825 | 3150 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_right_pocket | 1 | 79.87301587301587 | 79.92030759139313 | 79.87301587301587 | 3150 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_right_pocket | 1 | 67.52380952380952 | 64.44231024076534 | 67.52380952380952 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_right_pocket | 1 | 86.09523809523809 | 86.0728180541891 | 86.0952380952381 | 3150 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_right_pocket | 1 | 84.28571428571429 | 84.36757360168627 | 84.28571428571428 | 3150 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_right_pocket | 1 | 70.85714285714285 | 68.35257791041062 | 70.85714285714285 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_right_pocket | 1 | 89.04761904761904 | 88.99900982519837 | 89.04761904761904 | 3150 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_right_pocket | 1 | 85.77777777777777 | 85.81954653988349 | 85.77777777777777 | 3150 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_right_pocket | 1 | 74.47619047619047 | 72.49897442297566 | 74.47619047619048 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_right_pocket | 1 | 89.46031746031747 | 89.40484808245623 | 89.46031746031746 | 3150 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_right_pocket | 1 | 87.04761904761905 | 87.05788113811019 | 87.04761904761905 | 3150 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_right_pocket | 1 | 78.15873015873017 | 76.99984940462537 | 78.15873015873017 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_right_pocket | 1 | 89.65079365079364 | 89.59555413571826 | 89.65079365079364 | 3150 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_right_pocket | 1 | 88.25396825396825 | 88.20870012612325 | 88.25396825396827 | 3150 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_right_pocket | 1 | 82.15873015873015 | 81.55264862328907 | 82.15873015873017 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 128 | False | False | phone_right_pocket | 1 | 90.15873015873017 | 90.10537878028106 | 90.15873015873017 | 3150 | ok |
| limubert_x | 0.055 | prototype | 128 | False | False | phone_right_pocket | 1 | 88.92063492063492 | 88.87998260389557 | 88.92063492063491 | 3150 | ok |
| limubert_x | 0.055 | ridge | 128 | False | False | phone_right_pocket | 1 | 86.34920634920636 | 86.04445480870844 | 86.34920634920634 | 3150 | ok |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_left_pocket | 1 | 34.44444444444444 | 27.33066929048874 | 34.44444444444445 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_left_pocket | 1 | 70.82539682539682 | 71.19691361948011 | 70.82539682539684 | 3150 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_left_pocket | 1 | 70.82539682539682 | 71.19691361948011 | 70.82539682539684 | 3150 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_left_pocket | 1 | 60.15873015873016 | 56.66191191965401 | 60.15873015873016 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_left_pocket | 1 | 78.31746031746032 | 78.4838384860582 | 78.31746031746032 | 3150 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_left_pocket | 1 | 76.38095238095238 | 76.62464181477017 | 76.38095238095238 | 3150 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_left_pocket | 1 | 61.61904761904762 | 57.56877951630247 | 61.61904761904762 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_left_pocket | 1 | 82.38095238095238 | 82.49975642713754 | 82.38095238095238 | 3150 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_left_pocket | 1 | 81.77777777777779 | 81.87485465009196 | 81.77777777777779 | 3150 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_left_pocket | 1 | 64.28571428571429 | 60.52066731141884 | 64.28571428571429 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_left_pocket | 1 | 85.14285714285714 | 85.26480748836923 | 85.14285714285714 | 3150 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_left_pocket | 1 | 84.28571428571429 | 84.37025360441346 | 84.28571428571429 | 3150 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_left_pocket | 1 | 70.0952380952381 | 67.72591069807608 | 70.09523809523809 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_left_pocket | 1 | 86.19047619047619 | 86.34874239941828 | 86.19047619047619 | 3150 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_left_pocket | 1 | 85.46031746031746 | 85.49566676181907 | 85.46031746031747 | 3150 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_left_pocket | 1 | 76.57142857142857 | 75.10919016104462 | 76.57142857142857 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_left_pocket | 1 | 87.20634920634922 | 87.3322269817635 | 87.2063492063492 | 3150 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_left_pocket | 1 | 87.39682539682539 | 87.399730497839 | 87.3968253968254 | 3150 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_left_pocket | 1 | 81.9047619047619 | 80.96229294018904 | 81.9047619047619 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_left_pocket | 1 | 87.93650793650794 | 88.0425395547862 | 87.93650793650795 | 3150 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_left_pocket | 1 | 86.60317460317461 | 86.61184815741969 | 86.6031746031746 | 3150 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_left_pocket | 1 | 84.82539682539682 | 84.24249311291209 | 84.82539682539682 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 128 | False | False | phone_left_pocket | 1 | 88.44444444444444 | 88.56819029819411 | 88.44444444444444 | 3150 | ok |
| limubert_x | 0.055 | prototype | 128 | False | False | phone_left_pocket | 1 | 86.8888888888889 | 86.89271506536785 | 86.8888888888889 | 3150 | ok |
| limubert_x | 0.055 | ridge | 128 | False | False | phone_left_pocket | 1 | 86.66666666666667 | 86.22272114151365 | 86.66666666666666 | 3150 | ok |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_belt | 1 | 36.22222222222222 | 31.94729565348906 | 36.222222222222214 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_belt | 1 | 54.507936507936506 | 55.000711027414376 | 54.507936507936506 | 3150 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_belt | 1 | 54.507936507936506 | 55.000711027414376 | 54.507936507936506 | 3150 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_belt | 1 | 50.095238095238095 | 47.662173503679234 | 50.095238095238095 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_belt | 1 | 62.53968253968254 | 63.00882598787781 | 62.53968253968255 | 3150 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_belt | 1 | 54.47619047619048 | 55.347745574613484 | 54.47619047619048 | 3150 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_belt | 1 | 49.047619047619044 | 46.60332256671062 | 49.047619047619044 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_belt | 1 | 66.53968253968254 | 66.97735622791781 | 66.53968253968253 | 3150 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_belt | 1 | 54.57142857142857 | 55.47990487168478 | 54.57142857142857 | 3150 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_belt | 1 | 52.476190476190474 | 49.02090135340632 | 52.47619047619049 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_belt | 1 | 68.25396825396825 | 68.93846766105133 | 68.25396825396825 | 3150 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_belt | 1 | 60.253968253968246 | 61.01955388680168 | 60.253968253968246 | 3150 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_belt | 1 | 57.17460317460318 | 51.88620912827776 | 57.17460317460318 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_belt | 1 | 69.58730158730158 | 70.30720141503984 | 69.58730158730158 | 3150 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_belt | 1 | 63.01587301587301 | 63.58795600371421 | 63.015873015873005 | 3150 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_belt | 1 | 59.65079365079365 | 52.9538159654663 | 59.65079365079365 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_belt | 1 | 70.47619047619048 | 71.11243602918462 | 70.47619047619048 | 3150 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_belt | 1 | 67.2063492063492 | 67.63547567096062 | 67.2063492063492 | 3150 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_belt | 1 | 60.730158730158735 | 53.32307267141275 | 60.730158730158735 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_belt | 1 | 71.14285714285714 | 71.62224115315026 | 71.14285714285715 | 3150 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_belt | 1 | 68.47619047619048 | 68.80175826533741 | 68.47619047619048 | 3150 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_belt | 1 | 61.873015873015866 | 54.35633002706185 | 61.87301587301588 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 128 | False | False | phone_belt | 1 | 71.33333333333334 | 71.72445390542705 | 71.33333333333333 | 3150 | ok |
| limubert_x | 0.055 | prototype | 128 | False | False | phone_belt | 1 | 70.73015873015873 | 70.98625757533007 | 70.73015873015872 | 3150 | ok |
| limubert_x | 0.055 | ridge | 128 | False | False | phone_belt | 1 | 63.77777777777778 | 57.230641732631206 | 63.77777777777778 | 3150 | ok |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | watch_wrist_proxy | 1 | 22.476190476190478 | 19.167545060675444 | 22.476190476190474 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | watch_wrist_proxy | 1 | 64.63492063492063 | 64.28638390417902 | 64.63492063492063 | 3150 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | watch_wrist_proxy | 1 | 64.63492063492063 | 64.28638390417902 | 64.63492063492063 | 3150 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | watch_wrist_proxy | 1 | 59.079365079365076 | 56.5077388905055 | 59.079365079365076 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | watch_wrist_proxy | 1 | 70.06349206349206 | 69.93733929134704 | 70.06349206349205 | 3150 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | watch_wrist_proxy | 1 | 68.57142857142857 | 68.16297433425082 | 68.57142857142858 | 3150 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | watch_wrist_proxy | 1 | 63.46031746031746 | 61.278535212817786 | 63.46031746031747 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | watch_wrist_proxy | 1 | 74.19047619047619 | 74.23244182172076 | 74.19047619047619 | 3150 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | watch_wrist_proxy | 1 | 71.39682539682539 | 71.01166365439468 | 71.3968253968254 | 3150 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | watch_wrist_proxy | 1 | 65.9047619047619 | 64.06730262498024 | 65.9047619047619 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | watch_wrist_proxy | 1 | 78.57142857142857 | 78.73813642761823 | 78.57142857142858 | 3150 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | watch_wrist_proxy | 1 | 73.42857142857143 | 73.08120240799984 | 73.42857142857142 | 3150 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | watch_wrist_proxy | 1 | 70.82539682539682 | 69.60560021838654 | 70.82539682539684 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | watch_wrist_proxy | 1 | 80.38095238095238 | 80.59272838645975 | 80.38095238095238 | 3150 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | watch_wrist_proxy | 1 | 75.42857142857143 | 75.12400352133939 | 75.42857142857143 | 3150 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | watch_wrist_proxy | 1 | 75.77777777777777 | 75.07706715386736 | 75.77777777777777 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | watch_wrist_proxy | 1 | 81.2063492063492 | 81.39805668258464 | 81.20634920634922 | 3150 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | watch_wrist_proxy | 1 | 76.31746031746032 | 75.97439327134626 | 76.3174603174603 | 3150 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | watch_wrist_proxy | 1 | 79.52380952380952 | 79.15911858281439 | 79.52380952380952 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | watch_wrist_proxy | 1 | 82.41269841269842 | 82.59407625927395 | 82.41269841269842 | 3150 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | watch_wrist_proxy | 1 | 76.73015873015872 | 76.38320002565519 | 76.73015873015875 | 3150 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | watch_wrist_proxy | 1 | 82.06349206349206 | 81.71070850676524 | 82.06349206349206 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 128 | False | False | watch_wrist_proxy | 1 | 82.15873015873015 | 82.22170721463287 | 82.15873015873015 | 3150 | ok |
| limubert_x | 0.055 | prototype | 128 | False | False | watch_wrist_proxy | 1 | 76.76190476190476 | 76.43268836683247 | 76.76190476190477 | 3150 | ok |
| limubert_x | 0.055 | ridge | 128 | False | False | watch_wrist_proxy | 1 | 84.98412698412699 | 84.71081508965197 | 84.984126984127 | 3150 | ok |
| limubert_x | 0.055 | zero_support | 0 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 77.36507936507937 | 77.82380905395873 | 77.36507936507937 | 3150 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 77.36507936507937 | 77.82380905395873 | 77.36507936507937 | 3150 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 61.71428571428571 | 56.392976613176124 | 61.71428571428572 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 81.68253968253968 | 82.00300780953206 | 81.68253968253968 | 3150 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 81.42857142857143 | 81.75431525863345 | 81.42857142857143 | 3150 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 62.82539682539683 | 57.297372242329224 | 62.82539682539683 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 84.95238095238096 | 85.13022128665449 | 84.95238095238096 | 3150 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 84.88888888888889 | 85.15543651623736 | 84.88888888888889 | 3150 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 63.42857142857142 | 56.94250715564071 | 63.42857142857142 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 86.53968253968254 | 86.63571665344784 | 86.53968253968254 | 3150 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 87.52380952380953 | 87.75077498538027 | 87.52380952380952 | 3150 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 66.66666666666666 | 60.18296691170221 | 66.66666666666667 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 87.07936507936508 | 87.19575114287578 | 87.07936507936508 | 3150 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 87.55555555555556 | 87.80558962193864 | 87.55555555555557 | 3150 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 70.06349206349206 | 63.98958304190911 | 70.06349206349206 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 88.98412698412699 | 89.070014904261 | 88.98412698412699 | 3150 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 88.0 | 88.26415763191771 | 87.99999999999999 | 3150 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 74.12698412698413 | 69.98458382804522 | 74.12698412698413 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 89.42857142857143 | 89.48105937696671 | 89.42857142857143 | 3150 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 88.47619047619048 | 88.69993203271483 | 88.47619047619048 | 3150 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 79.9047619047619 | 78.13174051914471 | 79.90476190476191 | 3150 | ok |
| limubert_x | 0.055 | 1nn | 128 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 89.46031746031747 | 89.5264497799738 | 89.46031746031746 | 3150 | ok |
| limubert_x | 0.055 | prototype | 128 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 88.31746031746032 | 88.55358750122052 | 88.31746031746032 | 3150 | ok |
| limubert_x | 0.055 | ridge | 128 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 86.09523809523809 | 85.49247292794962 | 86.09523809523809 | 3150 | ok |

### usc_had
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | stream | n_devices | accuracy | f1_macro | balanced_accuracy | n_queries | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halo | 2.203 | halo-classifier | 0 | True | True | phone_hip | 1 | 36.89861375373743 | 34.78948352884439 | 42.99170399440904 | 7358 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_hip | 1 | 33.514541995107365 | 32.10059607147963 | 38.55886535871959 | 7358 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_hip | 1 | 47.82549605871161 | 50.12900353236759 | 50.00086626635374 | 7358 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_hip | 1 | 47.82549605871161 | 50.12900353236759 | 50.00086626635374 | 7358 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_hip | 1 | 47.961402555042135 | 49.820337349351114 | 50.3103435656003 | 7358 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_hip | 1 | 49.38842076651264 | 50.684841495463786 | 53.6481933318008 | 7358 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_hip | 1 | 48.545800489263385 | 50.44386178817454 | 50.33105547352817 | 7358 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_hip | 1 | 53.003533568904594 | 55.303339835393714 | 55.18642717445527 | 7358 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_hip | 1 | 53.982060342484374 | 56.27738413733847 | 56.21478934956167 | 7358 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_hip | 1 | 53.35689045936396 | 54.976091044881045 | 55.82274344009479 | 7358 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_hip | 1 | 52.473498233215544 | 53.487210381152785 | 56.830739515767284 | 7358 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_hip | 1 | 53.88692579505301 | 55.91111992229042 | 55.777772949397054 | 7358 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_hip | 1 | 57.52921989671106 | 59.424858834037884 | 59.31892578450252 | 7358 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_hip | 1 | 58.765969013318845 | 60.62915690351348 | 60.82489620840552 | 7358 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_hip | 1 | 58.956238108181566 | 60.193438937753264 | 61.102823520922925 | 7358 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_hip | 1 | 54.865452568632776 | 55.208874585630326 | 58.83565889040785 | 7358 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_hip | 1 | 58.77955966295188 | 60.356978818173154 | 60.30025495259876 | 7358 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_hip | 1 | 61.98695297635227 | 63.34164424282178 | 63.284283961975106 | 7358 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_hip | 1 | 60.818157107909755 | 62.6052115353095 | 62.91982801466097 | 7358 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_hip | 1 | 61.70154933405817 | 62.71979428348274 | 63.62951851262183 | 7358 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_hip | 1 | 57.692307692307686 | 57.76896015237292 | 61.31456626729509 | 7358 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_hip | 1 | 63.49551508562109 | 64.6995112765719 | 64.62969590929379 | 7358 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_hip | 1 | 66.02337591736885 | 66.90615370667848 | 66.80100980583724 | 7358 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_hip | 1 | 61.98695297635227 | 63.24888494962939 | 63.727290957990434 | 7358 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_hip | 1 | 64.69149225332971 | 65.24734469970178 | 66.19837119651778 | 7358 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_hip | 1 | 59.3775482468062 | 58.8460155330232 | 62.61244479193253 | 7358 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_hip | 1 | 66.30877955966295 | 66.97506015566942 | 66.91158110448808 | 7358 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_hip | 1 | 70.2364773036151 | 70.82471789958996 | 70.71981455143082 | 7358 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_hip | 1 | 63.69937483011688 | 64.69889131465976 | 65.17629766963822 | 7358 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_hip | 1 | 67.76297907039957 | 67.84516046042832 | 68.69689877193602 | 7358 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_hip | 1 | 60.587116064147864 | 59.78991788851997 | 63.88292116567488 | 7358 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_hip | 1 | 69.85593911388965 | 70.10457840502725 | 70.09686704645448 | 7358 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_hip | 1 | 73.70209296004349 | 73.65757022681989 | 73.61006805875695 | 7358 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_hip | 1 | 64.28377276433814 | 65.01556421335368 | 65.58991836175801 | 7358 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_hip | 1 | 71.07909758086437 | 70.69003042568185 | 71.49903403758763 | 7358 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_hip | 1 | 60.913291655341126 | 59.82755214716352 | 64.06045352112216 | 7358 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_hip | 1 | 72.0983963033433 | 72.0915670957366 | 72.05939383296904 | 7358 | ok |
| halo | 0.789 | 1nn | 128 | True | True | phone_hip | 1 | 76.7871704267464 | 76.20354410923795 | 76.18959887283455 | 7358 | ok |
| halo | 0.789 | prototype | 128 | True | True | phone_hip | 1 | 64.77303615112803 | 65.60854869799144 | 66.23544971378406 | 7358 | ok |
| halo | 0.789 | ridge | 128 | True | True | phone_hip | 1 | 73.32155477031802 | 72.72428030431284 | 73.58562029068221 | 7358 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | phone_hip | 1 | 61.796683881489535 | 60.35368518576932 | 64.8006055622881 | 7358 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | phone_hip | 1 | 73.17205762435445 | 72.8786661110219 | 72.97862315929814 | 7358 | ok |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_hip | 1 | 24.830116879586843 | 23.600626309810988 | 28.105387883745024 | 7358 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_hip | 1 | 30.61973362326719 | 29.938946468231492 | 30.521360678422262 | 7358 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_hip | 1 | 30.61973362326719 | 29.938946468231492 | 30.521360678422262 | 7358 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_hip | 1 | 31.013862462625713 | 30.504367742747508 | 31.08473770793621 | 7358 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_hip | 1 | 33.35145419951074 | 33.2768536494459 | 34.17462546586032 | 7358 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_hip | 1 | 33.215547703180206 | 33.07862238564996 | 33.79958896230379 | 7358 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_hip | 1 | 33.867898885566724 | 33.54396765514171 | 34.75883042286429 | 7358 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_hip | 1 | 37.5917368850231 | 37.97278557123063 | 38.536565455194534 | 7358 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_hip | 1 | 36.95297635226964 | 36.94801415935886 | 37.86900842727566 | 7358 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_hip | 1 | 38.882848600163086 | 38.225281754762655 | 39.8153045084817 | 7358 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_hip | 1 | 41.64175047567274 | 42.259663806024214 | 42.7663704057602 | 7358 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_hip | 1 | 39.65751562924707 | 39.910052301055664 | 41.08823916244136 | 7358 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_hip | 1 | 43.16390323457461 | 42.7622803067109 | 44.7568585090684 | 7358 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_hip | 1 | 44.57733079641207 | 45.62217147262093 | 46.250226120199954 | 7358 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_hip | 1 | 41.20684968741506 | 41.535480940087226 | 42.85808797861044 | 7358 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_hip | 1 | 45.89562381081816 | 45.63374411602215 | 47.7630884448125 | 7358 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_hip | 1 | 48.60016308779559 | 49.75526513199933 | 50.16911955514295 | 7358 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_hip | 1 | 43.8842076651264 | 44.32026022274271 | 45.93066431009745 | 7358 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_hip | 1 | 49.592280511008425 | 49.79060819661189 | 51.87362747889428 | 7358 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_hip | 1 | 53.13944006523512 | 54.18517561342405 | 54.71564572634138 | 7358 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_hip | 1 | 44.536558847512914 | 44.97286493332484 | 46.83960106826905 | 7358 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_hip | 1 | 53.11225876596901 | 53.37469804905609 | 55.42215381062239 | 7358 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | phone_hip | 1 | 56.21092688230498 | 57.21183231406197 | 57.6770574711526 | 7358 | ok |
| harnet | 4.49 | prototype | 128 | False | False | phone_hip | 1 | 45.419951073661316 | 45.751072030265675 | 48.07875151943814 | 7358 | ok |
| harnet | 4.49 | ridge | 128 | False | False | phone_hip | 1 | 55.88475129111171 | 56.56454815039752 | 58.28082513818905 | 7358 | ok |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_hip | 1 | 27.23566186463713 | 24.872857613066017 | 31.069594116994022 | 7358 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_hip | 1 | 38.32563196520793 | 40.76995816152218 | 40.128227606889624 | 7358 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_hip | 1 | 38.32563196520793 | 40.76995816152218 | 40.128227606889624 | 7358 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_hip | 1 | 39.18184289209024 | 39.88706431850878 | 40.86265611875601 | 7358 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_hip | 1 | 42.64745854851862 | 45.40969712364538 | 44.58765887730169 | 7358 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_hip | 1 | 40.391410709431916 | 43.63998221661298 | 42.80285171181307 | 7358 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_hip | 1 | 41.38352813264474 | 42.24591768713017 | 43.42907048844394 | 7358 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_hip | 1 | 49.78254960587116 | 51.80294536476081 | 51.16409593095367 | 7358 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_hip | 1 | 43.53085077466703 | 46.88066540941694 | 46.47488884000115 | 7358 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_hip | 1 | 43.3949442783365 | 44.70728736810896 | 45.9708403943932 | 7358 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_hip | 1 | 53.69665670019027 | 55.2556849566417 | 54.760311971957485 | 7358 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_hip | 1 | 45.718945365588475 | 48.481376458886146 | 48.158767633930104 | 7358 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_hip | 1 | 47.050829029627614 | 47.87846066514439 | 49.16981243116651 | 7358 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_hip | 1 | 57.95053003533569 | 59.25487292295286 | 59.001513336534806 | 7358 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_hip | 1 | 49.6874150584398 | 51.49662871044897 | 51.484683941722565 | 7358 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_hip | 1 | 52.052188094590925 | 51.97764258173366 | 53.4478231612921 | 7358 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_hip | 1 | 61.429736341397124 | 62.54759930140974 | 62.205320818626944 | 7358 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_hip | 1 | 51.970644196792605 | 53.114052434398786 | 53.364378409408275 | 7358 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_hip | 1 | 55.95270453927698 | 55.98692998450535 | 57.45749273223847 | 7358 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_hip | 1 | 66.07773851590106 | 67.08476451179504 | 66.7967517473048 | 7358 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_hip | 1 | 54.213101386246265 | 54.202073220446714 | 55.07311216796628 | 7358 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_hip | 1 | 58.208752378363684 | 57.98197526676908 | 59.899535759773634 | 7358 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | phone_hip | 1 | 68.48328350095134 | 69.20304338041015 | 68.98514818153659 | 7358 | ok |
| unimts | 68.61 | prototype | 128 | True | False | phone_hip | 1 | 55.09649361239467 | 54.636477678401505 | 55.70359431795053 | 7358 | ok |
| unimts | 68.61 | ridge | 128 | True | False | phone_hip | 1 | 61.416145691764065 | 61.42332668586589 | 63.37769003140351 | 7358 | ok |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_hip | 1 | 9.160097852677358 | 1.3985723771580345 | 8.333333333333332 | 7358 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_hip | 1 | 15.53411253057896 | 15.001660130320538 | 15.270273765426246 | 7358 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_hip | 1 | 15.53411253057896 | 15.001660130320538 | 15.270273765426246 | 7358 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_hip | 1 | 14.052731720576245 | 13.585743950038589 | 14.194891693445358 | 7358 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_hip | 1 | 17.51834737700462 | 17.02445547408734 | 17.21930273655597 | 7358 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_hip | 1 | 17.137809187279153 | 16.706496736805533 | 17.04034853394495 | 7358 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_hip | 1 | 14.705082902962763 | 14.236977129956143 | 15.106539009565525 | 7358 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_hip | 1 | 19.964664310954063 | 19.394203405604515 | 19.73020995124711 | 7358 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_hip | 1 | 17.695025822234303 | 17.0939828887332 | 17.615709152536837 | 7358 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_hip | 1 | 16.41750475672737 | 15.575943842400127 | 16.509798171326917 | 7358 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_hip | 1 | 22.574069040500135 | 21.92188755309622 | 22.008389456289716 | 7358 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_hip | 1 | 18.374558303886925 | 17.743191303770057 | 18.27964926072697 | 7358 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_hip | 1 | 18.646371296547976 | 17.119748521746725 | 18.302665454238127 | 7358 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_hip | 1 | 25.18347377004621 | 24.202117267293122 | 24.319955443931352 | 7358 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_hip | 1 | 19.3530850774667 | 18.50451227958765 | 19.389778801425827 | 7358 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_hip | 1 | 20.861647186735528 | 17.985948117283133 | 19.878300723379184 | 7358 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_hip | 1 | 27.56183745583039 | 26.43130695682947 | 26.424074195306872 | 7358 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_hip | 1 | 19.38026637673281 | 18.333781453795282 | 19.337027704589886 | 7358 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_hip | 1 | 23.267192171785812 | 19.27157206263479 | 21.557644791898454 | 7358 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_hip | 1 | 30.72845882033161 | 29.515834916722888 | 29.594160377277017 | 7358 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_hip | 1 | 20.154933405816795 | 18.638425193677584 | 20.168441282985995 | 7358 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_hip | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | 1nn | 128 | True | False | phone_hip | 1 | 32.38651807556401 | 31.026637703210447 | 31.234013143256313 | 7358 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | phone_hip | 1 | 20.49469964664311 | 18.409911255217306 | 20.74963318086693 | 7358 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | phone_hip | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_hip | 1 | 13.37319923892362 | 12.782475736349005 | 15.51737901530619 | 7358 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_hip | 1 | 47.18673552595814 | 44.60453548356429 | 44.39490979667138 | 7358 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_hip | 1 | 47.18673552595814 | 44.60453548356429 | 44.39490979667138 | 7358 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_hip | 1 | 43.70752921989671 | 41.05195104151861 | 43.32092667546572 | 7358 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_hip | 1 | 54.22669203587932 | 51.70932281412012 | 51.40231008712546 | 7358 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_hip | 1 | 53.72383799945637 | 50.998444948317726 | 50.819090489771455 | 7358 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_hip | 1 | 48.80402283229138 | 45.41424758340434 | 48.31637950497371 | 7358 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_hip | 1 | 61.07637945093776 | 58.349169222737885 | 57.90140434210777 | 7358 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_hip | 1 | 58.41261212285948 | 55.09281713412659 | 54.7223093413188 | 7358 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_hip | 1 | 53.58793150312585 | 49.432199833003274 | 52.54990883866463 | 7358 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_hip | 1 | 67.55911932590378 | 64.8993132987195 | 64.41238080376314 | 7358 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_hip | 1 | 60.57352541451482 | 57.60002576058096 | 57.206465501670046 | 7358 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_hip | 1 | 56.836096765425395 | 52.55154528377146 | 55.935601105480146 | 7358 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_hip | 1 | 73.09051372655613 | 70.41790375384721 | 69.92761313961357 | 7358 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_hip | 1 | 62.557760260940476 | 59.403782512344186 | 58.966380229029184 | 7358 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_hip | 1 | 60.1793965751563 | 56.012709434047444 | 59.23732017951663 | 7358 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_hip | 1 | 76.7871704267464 | 74.28954243487672 | 73.76753418114997 | 7358 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_hip | 1 | 64.18863821690677 | 60.653303049038634 | 60.298601302128596 | 7358 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_hip | 1 | 62.245175319380266 | 58.31874947991157 | 61.384732529405184 | 7358 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_hip | 1 | 80.21201413427562 | 77.67651429674972 | 77.28383595647115 | 7358 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_hip | 1 | 64.84098939929329 | 61.20491686006041 | 60.8149696005458 | 7358 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_hip | 1 | 63.82169067681435 | 60.10531184819376 | 62.75432321998491 | 7358 | ok |
| limubert_x | 0.055 | 1nn | 128 | False | False | phone_hip | 1 | 82.22343027996737 | 79.66790673444383 | 79.31591922415838 | 7358 | ok |
| limubert_x | 0.055 | prototype | 128 | False | False | phone_hip | 1 | 65.12639304158738 | 61.60899472061325 | 61.24746267116728 | 7358 | ok |
| limubert_x | 0.055 | ridge | 128 | False | False | phone_hip | 1 | 65.64283772764338 | 62.2713538115902 | 64.46218861559531 | 7358 | ok |

### ut_complex
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | stream | n_devices | accuracy | f1_macro | balanced_accuracy | n_queries | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halo | 2.203 | halo-classifier | 0 | True | True | watch_wrist | 1 | 42.05128205128205 | 38.061061964148415 | 42.05128205128205 | 5850 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | watch_wrist | 1 | 25.572649572649574 | 22.976145212212806 | 25.572649572649574 | 5850 | ok |
| halo | 0.789 | 1nn | 1 | True | True | watch_wrist | 1 | 56.940170940170944 | 56.24736431775733 | 56.940170940170944 | 5850 | ok |
| halo | 0.789 | prototype | 1 | True | True | watch_wrist | 1 | 56.940170940170944 | 56.24736431775733 | 56.940170940170944 | 5850 | ok |
| halo | 0.789 | ridge | 1 | True | True | watch_wrist | 1 | 58.05128205128205 | 57.13969665272434 | 58.05128205128205 | 5850 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | watch_wrist | 1 | 60.136752136752136 | 58.81506862995646 | 60.13675213675215 | 5850 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | watch_wrist | 1 | 56.991452991452995 | 56.0864116553613 | 56.991452991452995 | 5850 | ok |
| halo | 0.789 | 1nn | 2 | True | True | watch_wrist | 1 | 62.581196581196586 | 62.25266730554135 | 62.581196581196586 | 5850 | ok |
| halo | 0.789 | prototype | 2 | True | True | watch_wrist | 1 | 65.43589743589745 | 64.9973727360313 | 65.43589743589743 | 5850 | ok |
| halo | 0.789 | ridge | 2 | True | True | watch_wrist | 1 | 65.91452991452992 | 65.15292180106225 | 65.91452991452992 | 5850 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | watch_wrist | 1 | 64.82051282051282 | 63.863252912863956 | 64.8205128205128 | 5850 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | watch_wrist | 1 | 63.863247863247864 | 63.36360012159984 | 63.863247863247864 | 5850 | ok |
| halo | 0.789 | 1nn | 4 | True | True | watch_wrist | 1 | 67.86324786324786 | 67.5840493978065 | 67.86324786324786 | 5850 | ok |
| halo | 0.789 | prototype | 4 | True | True | watch_wrist | 1 | 70.52991452991452 | 70.12001793906664 | 70.52991452991452 | 5850 | ok |
| halo | 0.789 | ridge | 4 | True | True | watch_wrist | 1 | 71.84615384615385 | 71.00405510214401 | 71.84615384615385 | 5850 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | watch_wrist | 1 | 69.14529914529915 | 68.34247351551099 | 69.14529914529915 | 5850 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | watch_wrist | 1 | 69.16239316239316 | 68.5466576775232 | 69.16239316239316 | 5850 | ok |
| halo | 0.789 | 1nn | 8 | True | True | watch_wrist | 1 | 70.25641025641025 | 70.04651881964651 | 70.25641025641026 | 5850 | ok |
| halo | 0.789 | prototype | 8 | True | True | watch_wrist | 1 | 73.8119658119658 | 73.42336351232898 | 73.8119658119658 | 5850 | ok |
| halo | 0.789 | ridge | 8 | True | True | watch_wrist | 1 | 74.71794871794873 | 73.95933383082323 | 74.71794871794872 | 5850 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | watch_wrist | 1 | 71.33333333333334 | 70.65592401283303 | 71.33333333333334 | 5850 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | watch_wrist | 1 | 72.64957264957265 | 72.00236274346832 | 72.64957264957265 | 5850 | ok |
| halo | 0.789 | 1nn | 16 | True | True | watch_wrist | 1 | 72.75213675213675 | 72.57896726332629 | 72.75213675213675 | 5850 | ok |
| halo | 0.789 | prototype | 16 | True | True | watch_wrist | 1 | 76.13675213675214 | 75.832282982534 | 76.13675213675212 | 5850 | ok |
| halo | 0.789 | ridge | 16 | True | True | watch_wrist | 1 | 77.40170940170941 | 76.7587445799395 | 77.4017094017094 | 5850 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | watch_wrist | 1 | 73.12820512820512 | 72.58491650881292 | 73.12820512820512 | 5850 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | watch_wrist | 1 | 75.2991452991453 | 74.56577251596042 | 75.29914529914531 | 5850 | ok |
| halo | 0.789 | 1nn | 32 | True | True | watch_wrist | 1 | 75.93162393162393 | 75.8201652297947 | 75.93162393162393 | 5850 | ok |
| halo | 0.789 | prototype | 32 | True | True | watch_wrist | 1 | 77.17948717948718 | 76.99286539093951 | 77.17948717948718 | 5850 | ok |
| halo | 0.789 | ridge | 32 | True | True | watch_wrist | 1 | 79.47008547008548 | 78.93644721110105 | 79.47008547008548 | 5850 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | watch_wrist | 1 | 74.06837606837607 | 73.60012622182145 | 74.06837606837608 | 5850 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | watch_wrist | 1 | 78.49572649572649 | 77.91925178676802 | 78.49572649572649 | 5850 | ok |
| halo | 0.789 | 1nn | 64 | True | True | watch_wrist | 1 | 77.36752136752136 | 77.25828237408055 | 77.36752136752136 | 5850 | ok |
| halo | 0.789 | prototype | 64 | True | True | watch_wrist | 1 | 77.33333333333333 | 77.07324099270785 | 77.33333333333334 | 5850 | ok |
| halo | 0.789 | ridge | 64 | True | True | watch_wrist | 1 | 80.76923076923077 | 80.42043437309022 | 80.76923076923077 | 5850 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | watch_wrist | 1 | 74.66666666666667 | 74.23744988213403 | 74.66666666666667 | 5850 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | watch_wrist | 1 | 78.97435897435898 | 78.41845358521446 | 78.97435897435898 | 5850 | ok |
| halo | 0.789 | 1nn | 128 | True | True | watch_wrist | 1 | 78.63247863247864 | 78.54013875591546 | 78.63247863247864 | 5850 | ok |
| halo | 0.789 | prototype | 128 | True | True | watch_wrist | 1 | 77.60683760683762 | 77.34881414639922 | 77.60683760683762 | 5850 | ok |
| halo | 0.789 | ridge | 128 | True | True | watch_wrist | 1 | 82.11965811965813 | 81.82575238000986 | 82.11965811965813 | 5850 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | watch_wrist | 1 | 74.85470085470085 | 74.45121947256409 | 74.85470085470087 | 5850 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | watch_wrist | 1 | 80.22222222222221 | 79.65325182525021 | 80.22222222222221 | 5850 | ok |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | watch_wrist | 1 | 36.73504273504273 | 33.98719145981367 | 36.73504273504273 | 5850 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | watch_wrist | 1 | 40.991452991452995 | 39.94491868303169 | 40.991452991452995 | 5850 | ok |
| harnet | 4.49 | prototype | 1 | False | False | watch_wrist | 1 | 40.991452991452995 | 39.94491868303169 | 40.991452991452995 | 5850 | ok |
| harnet | 4.49 | ridge | 1 | False | False | watch_wrist | 1 | 41.521367521367516 | 40.15767509552812 | 41.521367521367516 | 5850 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | watch_wrist | 1 | 47.53846153846154 | 46.5867671423268 | 47.53846153846153 | 5850 | ok |
| harnet | 4.49 | prototype | 2 | False | False | watch_wrist | 1 | 46.51282051282051 | 45.29103664124165 | 46.51282051282051 | 5850 | ok |
| harnet | 4.49 | ridge | 2 | False | False | watch_wrist | 1 | 46.547008547008545 | 44.80755404387859 | 46.547008547008545 | 5850 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | watch_wrist | 1 | 51.24786324786324 | 50.533791898543925 | 51.24786324786326 | 5850 | ok |
| harnet | 4.49 | prototype | 4 | False | False | watch_wrist | 1 | 51.042735042735046 | 49.953472672796025 | 51.042735042735046 | 5850 | ok |
| harnet | 4.49 | ridge | 4 | False | False | watch_wrist | 1 | 52.78632478632479 | 51.15853579528914 | 52.78632478632479 | 5850 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | watch_wrist | 1 | 53.880341880341874 | 53.39817296010195 | 53.880341880341895 | 5850 | ok |
| harnet | 4.49 | prototype | 8 | False | False | watch_wrist | 1 | 54.837606837606835 | 53.92216631645114 | 54.83760683760684 | 5850 | ok |
| harnet | 4.49 | ridge | 8 | False | False | watch_wrist | 1 | 59.504273504273506 | 58.19549743650273 | 59.5042735042735 | 5850 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | watch_wrist | 1 | 57.948717948717956 | 57.58358244334511 | 57.948717948717956 | 5850 | ok |
| harnet | 4.49 | prototype | 16 | False | False | watch_wrist | 1 | 57.55555555555556 | 56.6431160840734 | 57.55555555555556 | 5850 | ok |
| harnet | 4.49 | ridge | 16 | False | False | watch_wrist | 1 | 65.02564102564102 | 63.70137666689947 | 65.02564102564102 | 5850 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | watch_wrist | 1 | 60.581196581196586 | 60.32372522304962 | 60.581196581196586 | 5850 | ok |
| harnet | 4.49 | prototype | 32 | False | False | watch_wrist | 1 | 58.82051282051282 | 57.92981930574664 | 58.82051282051282 | 5850 | ok |
| harnet | 4.49 | ridge | 32 | False | False | watch_wrist | 1 | 69.88034188034187 | 68.77087431298656 | 69.88034188034187 | 5850 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | watch_wrist | 1 | 63.19658119658119 | 63.141462276134106 | 63.19658119658118 | 5850 | ok |
| harnet | 4.49 | prototype | 64 | False | False | watch_wrist | 1 | 59.98290598290599 | 59.23059143106363 | 59.98290598290599 | 5850 | ok |
| harnet | 4.49 | ridge | 64 | False | False | watch_wrist | 1 | 73.6068376068376 | 72.70615827646944 | 73.60683760683759 | 5850 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | watch_wrist | 1 | 64.23931623931624 | 64.32613604221899 | 64.23931623931624 | 5850 | ok |
| harnet | 4.49 | prototype | 128 | False | False | watch_wrist | 1 | 61.21367521367521 | 60.463996630563635 | 61.21367521367521 | 5850 | ok |
| harnet | 4.49 | ridge | 128 | False | False | watch_wrist | 1 | 75.74358974358975 | 74.9965966607413 | 75.74358974358974 | 5850 | ok |
| unimts | 68.61 | native_zero_support | 0 | True | False | watch_wrist | 1 | 27.726495726495727 | 21.81291237185709 | 27.726495726495727 | 5850 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | watch_wrist | 1 | 43.53846153846154 | 42.61187228008409 | 43.53846153846154 | 5850 | ok |
| unimts | 68.61 | prototype | 1 | True | False | watch_wrist | 1 | 43.53846153846154 | 42.61187228008409 | 43.53846153846154 | 5850 | ok |
| unimts | 68.61 | ridge | 1 | True | False | watch_wrist | 1 | 39.863247863247864 | 38.459295827729036 | 39.86324786324787 | 5850 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | watch_wrist | 1 | 48.495726495726494 | 47.71894876301756 | 48.495726495726494 | 5850 | ok |
| unimts | 68.61 | prototype | 2 | True | False | watch_wrist | 1 | 48.01709401709402 | 47.209013326103474 | 48.01709401709402 | 5850 | ok |
| unimts | 68.61 | ridge | 2 | True | False | watch_wrist | 1 | 44.34188034188034 | 43.242964620211566 | 44.34188034188034 | 5850 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | watch_wrist | 1 | 52.85470085470085 | 52.33939850526973 | 52.85470085470085 | 5850 | ok |
| unimts | 68.61 | prototype | 4 | True | False | watch_wrist | 1 | 51.67521367521367 | 50.78385299393891 | 51.67521367521368 | 5850 | ok |
| unimts | 68.61 | ridge | 4 | True | False | watch_wrist | 1 | 48.5982905982906 | 47.19008673061671 | 48.59829059829059 | 5850 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | watch_wrist | 1 | 56.06837606837607 | 55.75700834310966 | 56.06837606837607 | 5850 | ok |
| unimts | 68.61 | prototype | 8 | True | False | watch_wrist | 1 | 54.68376068376068 | 53.73825914950927 | 54.683760683760674 | 5850 | ok |
| unimts | 68.61 | ridge | 8 | True | False | watch_wrist | 1 | 52.547008547008545 | 51.071198590945265 | 52.54700854700853 | 5850 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | watch_wrist | 1 | 58.42735042735042 | 58.252942045857296 | 58.42735042735042 | 5850 | ok |
| unimts | 68.61 | prototype | 16 | True | False | watch_wrist | 1 | 56.44444444444444 | 55.56721971514988 | 56.44444444444445 | 5850 | ok |
| unimts | 68.61 | ridge | 16 | True | False | watch_wrist | 1 | 56.30769230769231 | 54.806237280314896 | 56.307692307692314 | 5850 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | watch_wrist | 1 | 60.957264957264954 | 60.9333355183823 | 60.95726495726497 | 5850 | ok |
| unimts | 68.61 | prototype | 32 | True | False | watch_wrist | 1 | 58.37606837606838 | 57.084814327477396 | 58.37606837606839 | 5850 | ok |
| unimts | 68.61 | ridge | 32 | True | False | watch_wrist | 1 | 60.99145299145299 | 59.340589349983105 | 60.99145299145299 | 5850 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | watch_wrist | 1 | 63.572649572649574 | 63.65532745882832 | 63.572649572649574 | 5850 | ok |
| unimts | 68.61 | prototype | 64 | True | False | watch_wrist | 1 | 59.726495726495735 | 58.40281757951809 | 59.726495726495735 | 5850 | ok |
| unimts | 68.61 | ridge | 64 | True | False | watch_wrist | 1 | 65.41880341880342 | 63.86595402717141 | 65.41880341880344 | 5850 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | watch_wrist | 1 | 65.24786324786325 | 65.24758717698381 | 65.24786324786326 | 5850 | ok |
| unimts | 68.61 | prototype | 128 | True | False | watch_wrist | 1 | 61.72649572649572 | 60.03233812328167 | 61.726495726495735 | 5850 | ok |
| unimts | 68.61 | ridge | 128 | True | False | watch_wrist | 1 | 69.07692307692308 | 67.70752345180246 | 69.07692307692308 | 5850 | ok |
| normwear | 1293.86 | native_zero_support | 0 | True | False | watch_wrist | 1 | 7.6923076923076925 | 1.0989010989010988 | 7.6923076923076925 | 5850 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | watch_wrist | 1 | 18.205128205128204 | 18.194480041682866 | 18.205128205128204 | 5850 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | watch_wrist | 1 | 18.22222222222222 | 18.20970685707039 | 18.22222222222222 | 5850 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | watch_wrist | 1 | 15.11111111111111 | 14.33275167350879 | 15.111111111111114 | 5850 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | watch_wrist | 1 | 22.47863247863248 | 22.48199424604747 | 22.478632478632477 | 5850 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | watch_wrist | 1 | 19.726495726495727 | 19.34868558356763 | 19.726495726495724 | 5850 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | watch_wrist | 1 | 16.153846153846153 | 14.594716124857818 | 16.153846153846153 | 5850 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | watch_wrist | 1 | 25.521367521367523 | 25.492041801536747 | 25.521367521367523 | 5850 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | watch_wrist | 1 | 20.752136752136753 | 20.186484689238853 | 20.752136752136753 | 5850 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | watch_wrist | 1 | 17.47008547008547 | 14.671585233786843 | 17.470085470085472 | 5850 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | watch_wrist | 1 | 28.786324786324784 | 28.93195411901605 | 28.786324786324784 | 5850 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | watch_wrist | 1 | 21.145299145299145 | 20.0128876454331 | 21.14529914529915 | 5850 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | watch_wrist | 1 | 19.042735042735043 | 14.817238900749002 | 19.042735042735043 | 5850 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | watch_wrist | 1 | 32.75213675213675 | 32.77117196712391 | 32.75213675213676 | 5850 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | watch_wrist | 1 | 21.863247863247864 | 19.82881689958724 | 21.86324786324786 | 5850 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | watch_wrist | 1 | 22.0 | 16.0393365704736 | 21.999999999999993 | 5850 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | watch_wrist | 1 | 35.98290598290598 | 36.12412180107701 | 35.982905982905976 | 5850 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | watch_wrist | 1 | 22.803418803418804 | 19.86719048143898 | 22.8034188034188 | 5850 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | watch_wrist | 1 | 26.256410256410255 | 18.72987518770775 | 26.256410256410255 | 5850 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | watch_wrist | 1 | 38.78632478632479 | 38.89348883439586 | 38.78632478632478 | 5850 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | watch_wrist | 1 | 22.923076923076923 | 19.712296898289125 | 22.92307692307692 | 5850 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | watch_wrist | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | 1nn | 128 | True | False | watch_wrist | 1 | 41.65811965811966 | 41.59728925131112 | 41.658119658119666 | 5850 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | watch_wrist | 1 | 22.444444444444443 | 18.89735900999499 | 22.444444444444446 | 5850 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | watch_wrist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | watch_wrist | 1 | 10.307692307692308 | 9.623060187562508 | 10.307692307692308 | 5850 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | watch_wrist | 1 | 50.34188034188034 | 48.9147519659143 | 50.34188034188034 | 5850 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | watch_wrist | 1 | 50.34188034188034 | 48.9147519659143 | 50.34188034188034 | 5850 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | watch_wrist | 1 | 43.24786324786325 | 39.21510814540214 | 43.24786324786325 | 5850 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | watch_wrist | 1 | 55.77777777777778 | 54.959091964309515 | 55.77777777777778 | 5850 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | watch_wrist | 1 | 54.08547008547009 | 52.42692838531833 | 54.08547008547009 | 5850 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | watch_wrist | 1 | 45.2991452991453 | 40.17080576941021 | 45.2991452991453 | 5850 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | watch_wrist | 1 | 61.26495726495727 | 60.728207873715824 | 61.264957264957275 | 5850 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | watch_wrist | 1 | 57.965811965811966 | 56.20658260970356 | 57.965811965811966 | 5850 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | watch_wrist | 1 | 47.8974358974359 | 42.26579725372584 | 47.89743589743589 | 5850 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | watch_wrist | 1 | 65.72649572649573 | 65.53833797268645 | 65.72649572649573 | 5850 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | watch_wrist | 1 | 60.239316239316246 | 58.41530321081056 | 60.23931623931623 | 5850 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | watch_wrist | 1 | 51.70940170940172 | 45.945239732061374 | 51.70940170940172 | 5850 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | watch_wrist | 1 | 69.6068376068376 | 69.50084771129875 | 69.60683760683762 | 5850 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | watch_wrist | 1 | 62.66666666666667 | 60.861846142576034 | 62.66666666666667 | 5850 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | watch_wrist | 1 | 55.69230769230769 | 49.782775054901784 | 55.692307692307686 | 5850 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | watch_wrist | 1 | 72.61538461538461 | 72.52080642966607 | 72.61538461538461 | 5850 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | watch_wrist | 1 | 62.76923076923077 | 60.74230799708824 | 62.76923076923077 | 5850 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | watch_wrist | 1 | 59.641025641025635 | 54.16967574367006 | 59.64102564102565 | 5850 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | watch_wrist | 1 | 74.17094017094017 | 74.04029993356727 | 74.17094017094018 | 5850 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | watch_wrist | 1 | 63.401709401709404 | 61.261086429949266 | 63.401709401709404 | 5850 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | watch_wrist | 1 | 63.2991452991453 | 58.482316958966464 | 63.2991452991453 | 5850 | ok |
| limubert_x | 0.055 | 1nn | 128 | False | False | watch_wrist | 1 | 76.29059829059828 | 76.21612148623798 | 76.29059829059828 | 5850 | ok |
| limubert_x | 0.055 | prototype | 128 | False | False | watch_wrist | 1 | 63.67521367521367 | 61.460981826686776 | 63.67521367521368 | 5850 | ok |
| limubert_x | 0.055 | ridge | 128 | False | False | watch_wrist | 1 | 66.82051282051282 | 62.86674229542173 | 66.82051282051282 | 5850 | ok |

### Dataset-Balanced Mean
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | accuracy | f1_macro | n_datasets |
|---|---|---|---|---|---|---|---|---|
| halo | 0.789 | 1nn | 1 | True | True | 56.335 | 55.574 | 6 |
| halo | 0.789 | 1nn | 2 | True | True | 60.802 | 60.274 | 6 |
| halo | 0.789 | 1nn | 4 | True | True | 64.488 | 64.029 | 6 |
| halo | 0.789 | 1nn | 8 | True | True | 66.633 | 66.267 | 6 |
| halo | 0.789 | 1nn | 16 | True | True | 68.709 | 68.295 | 6 |
| halo | 0.789 | 1nn | 32 | True | True | 70.511 | 70.158 | 6 |
| halo | 0.789 | 1nn | 64 | True | True | 71.809 | 70.975 | 6 |
| halo | 0.789 | 1nn | 128 | True | True | 72.984 | 72.235 | 6 |
| halo | 2.203 | halo-classifier | 0 | True | True | 51.934 | 48.884 | 6 |
| halo | 2.203 | halo-classifier | 1 | True | True | 59.261 | 58.082 | 6 |
| halo | 2.203 | halo-classifier | 2 | True | True | 62.86 | 61.753 | 6 |
| halo | 2.203 | halo-classifier | 4 | True | True | 65.419 | 64.333 | 6 |
| halo | 2.203 | halo-classifier | 8 | True | True | 67.17 | 66.091 | 6 |
| halo | 2.203 | halo-classifier | 16 | True | True | 68.289 | 67.157 | 6 |
| halo | 2.203 | halo-classifier | 32 | True | True | 68.667 | 67.558 | 6 |
| halo | 2.203 | halo-classifier | 64 | True | True | 69.228 | 67.693 | 6 |
| halo | 2.203 | halo-classifier | 128 | True | True | 69.5 | 67.696 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | 56.086 | 55.083 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | 61.354 | 60.641 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | 65.57 | 64.892 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | 68.484 | 67.911 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | 70.934 | 70.338 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | 72.539 | 72.009 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | 73.595 | 72.597 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | 74.454 | 73.665 | 6 |
| halo | 0.789 | prototype | 1 | True | True | 56.335 | 55.574 | 6 |
| halo | 0.789 | prototype | 2 | True | True | 61.838 | 61.034 | 6 |
| halo | 0.789 | prototype | 4 | True | True | 66.386 | 65.636 | 6 |
| halo | 0.789 | prototype | 8 | True | True | 68.822 | 68.129 | 6 |
| halo | 0.789 | prototype | 16 | True | True | 70.201 | 69.483 | 6 |
| halo | 0.789 | prototype | 32 | True | True | 71.262 | 70.605 | 6 |
| halo | 0.789 | prototype | 64 | True | True | 71.61 | 70.413 | 6 |
| halo | 0.789 | prototype | 128 | True | True | 71.965 | 71.118 | 6 |
| halo | 0.789 | ridge | 1 | True | True | 56.798 | 55.784 | 6 |
| halo | 0.789 | ridge | 2 | True | True | 62.436 | 61.328 | 6 |
| halo | 0.789 | ridge | 4 | True | True | 66.967 | 65.929 | 6 |
| halo | 0.789 | ridge | 8 | True | True | 69.862 | 69.005 | 6 |
| halo | 0.789 | ridge | 16 | True | True | 72.226 | 71.435 | 6 |
| halo | 0.789 | ridge | 32 | True | True | 73.662 | 72.946 | 6 |
| halo | 0.789 | ridge | 64 | True | True | 75.072 | 73.867 | 6 |
| halo | 0.789 | ridge | 128 | True | True | 76.278 | 75.279 | 6 |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | 40.016 | 35.39 | 6 |
| harnet | 4.49 | 1nn | 1 | False | False | 41.719 | 40.137 | 6 |
| harnet | 4.49 | 1nn | 2 | False | False | 45.741 | 44.309 | 6 |
| harnet | 4.49 | 1nn | 4 | False | False | 48.815 | 47.733 | 6 |
| harnet | 4.49 | 1nn | 8 | False | False | 51.427 | 50.602 | 6 |
| harnet | 4.49 | 1nn | 16 | False | False | 54.25 | 53.643 | 6 |
| harnet | 4.49 | 1nn | 32 | False | False | 56.45 | 56.07 | 6 |
| harnet | 4.49 | 1nn | 64 | False | False | 58.922 | 58.364 | 6 |
| harnet | 4.49 | 1nn | 128 | False | False | 60.612 | 60.166 | 6 |
| harnet | 4.49 | prototype | 1 | False | False | 41.719 | 40.137 | 6 |
| harnet | 4.49 | prototype | 2 | False | False | 45.39 | 43.971 | 6 |
| harnet | 4.49 | prototype | 4 | False | False | 48.885 | 47.629 | 6 |
| harnet | 4.49 | prototype | 8 | False | False | 51.86 | 50.872 | 6 |
| harnet | 4.49 | prototype | 16 | False | False | 53.541 | 52.557 | 6 |
| harnet | 4.49 | prototype | 32 | False | False | 54.983 | 54.134 | 6 |
| harnet | 4.49 | prototype | 64 | False | False | 55.53 | 54.385 | 6 |
| harnet | 4.49 | prototype | 128 | False | False | 56.224 | 55.278 | 6 |
| harnet | 4.49 | ridge | 1 | False | False | 41.971 | 40.224 | 6 |
| harnet | 4.49 | ridge | 2 | False | False | 45.803 | 43.965 | 6 |
| harnet | 4.49 | ridge | 4 | False | False | 50.191 | 48.456 | 6 |
| harnet | 4.49 | ridge | 8 | False | False | 54.935 | 53.611 | 6 |
| harnet | 4.49 | ridge | 16 | False | False | 58.207 | 57.143 | 6 |
| harnet | 4.49 | ridge | 32 | False | False | 61.77 | 61.161 | 6 |
| harnet | 4.49 | ridge | 64 | False | False | 64.361 | 63.535 | 6 |
| harnet | 4.49 | ridge | 128 | False | False | 66.975 | 66.309 | 6 |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | 37.191 | 33.785 | 6 |
| limubert_x | 0.055 | 1nn | 1 | False | False | 48.638 | 47.313 | 5 |
| limubert_x | 0.055 | 1nn | 2 | False | False | 54.736 | 53.394 | 5 |
| limubert_x | 0.055 | 1nn | 4 | False | False | 59.749 | 58.436 | 5 |
| limubert_x | 0.055 | 1nn | 8 | False | False | 63.845 | 62.666 | 5 |
| limubert_x | 0.055 | 1nn | 16 | False | False | 66.792 | 65.735 | 5 |
| limubert_x | 0.055 | 1nn | 32 | False | False | 69.021 | 68.122 | 5 |
| limubert_x | 0.055 | 1nn | 64 | False | False | 70.659 | 69.832 | 5 |
| limubert_x | 0.055 | 1nn | 128 | False | False | 71.792 | 70.996 | 5 |
| limubert_x | 0.055 | prototype | 1 | False | False | 48.638 | 47.313 | 5 |
| limubert_x | 0.055 | prototype | 2 | False | False | 53.21 | 51.828 | 5 |
| limubert_x | 0.055 | prototype | 4 | False | False | 57.063 | 55.516 | 5 |
| limubert_x | 0.055 | prototype | 8 | False | False | 59.412 | 57.818 | 5 |
| limubert_x | 0.055 | prototype | 16 | False | False | 60.926 | 59.169 | 5 |
| limubert_x | 0.055 | prototype | 32 | False | False | 62.011 | 59.946 | 5 |
| limubert_x | 0.055 | prototype | 64 | False | False | 62.255 | 60.058 | 5 |
| limubert_x | 0.055 | prototype | 128 | False | False | 62.318 | 59.923 | 5 |
| limubert_x | 0.055 | ridge | 1 | False | False | 44.125 | 41.035 | 5 |
| limubert_x | 0.055 | ridge | 2 | False | False | 47.271 | 43.741 | 5 |
| limubert_x | 0.055 | ridge | 4 | False | False | 50.624 | 46.628 | 5 |
| limubert_x | 0.055 | ridge | 8 | False | False | 54.313 | 50.041 | 5 |
| limubert_x | 0.055 | ridge | 16 | False | False | 57.378 | 52.731 | 5 |
| limubert_x | 0.055 | ridge | 32 | False | False | 60.344 | 55.604 | 5 |
| limubert_x | 0.055 | ridge | 64 | False | False | 62.444 | 57.871 | 5 |
| limubert_x | 0.055 | ridge | 128 | False | False | 64.787 | 60.742 | 5 |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | 27.409 | 23.925 | 5 |
| normwear | 1293.86 | 1nn | 1 | True | False | 21.667 | 20.964 | 6 |
| normwear | 1293.86 | 1nn | 2 | True | False | 23.564 | 22.876 | 6 |
| normwear | 1293.86 | 1nn | 4 | True | False | 26.094 | 25.354 | 6 |
| normwear | 1293.86 | 1nn | 8 | True | False | 28.933 | 28.158 | 6 |
| normwear | 1293.86 | 1nn | 16 | True | False | 31.591 | 30.743 | 6 |
| normwear | 1293.86 | 1nn | 32 | True | False | 34.102 | 33.187 | 6 |
| normwear | 1293.86 | 1nn | 64 | True | False | 36.761 | 35.759 | 6 |
| normwear | 1293.86 | 1nn | 128 | True | False | 39.142 | 38.104 | 6 |
| normwear | 1293.86 | native_zero_support | 0 | True | False | 14.624 | 3.45 | 6 |
| normwear | 1293.86 | prototype | 1 | True | False | 21.67 | 20.966 | 6 |
| normwear | 1293.86 | prototype | 2 | True | False | 22.901 | 22.007 | 6 |
| normwear | 1293.86 | prototype | 4 | True | False | 24.124 | 23.005 | 6 |
| normwear | 1293.86 | prototype | 8 | True | False | 24.57 | 23.079 | 6 |
| normwear | 1293.86 | prototype | 16 | True | False | 25.4 | 23.543 | 6 |
| normwear | 1293.86 | prototype | 32 | True | False | 25.465 | 23.114 | 6 |
| normwear | 1293.86 | prototype | 64 | True | False | 25.524 | 22.639 | 6 |
| normwear | 1293.86 | prototype | 128 | True | False | 25.939 | 22.448 | 6 |
| normwear | 1293.86 | ridge | 1 | True | False | 20.306 | 19.079 | 6 |
| normwear | 1293.86 | ridge | 2 | True | False | 21.233 | 19.624 | 6 |
| normwear | 1293.86 | ridge | 4 | True | False | 22.661 | 20.356 | 6 |
| normwear | 1293.86 | ridge | 8 | True | False | 23.558 | 20.378 | 6 |
| normwear | 1293.86 | ridge | 16 | True | False | 25.152 | 20.856 | 6 |
| normwear | 1293.86 | ridge | 32 | True | False | 26.715 | 21.717 | 6 |
| normwear | 1293.86 | ridge | 64 | True | False | 29.536 | 24.707 | 4 |
| unimts | 68.61 | 1nn | 1 | True | False | 50.944 | 50.609 | 6 |
| unimts | 68.61 | 1nn | 2 | True | False | 56.571 | 56.47 | 6 |
| unimts | 68.61 | 1nn | 4 | True | False | 61.223 | 61.154 | 6 |
| unimts | 68.61 | 1nn | 8 | True | False | 63.969 | 63.946 | 6 |
| unimts | 68.61 | 1nn | 16 | True | False | 65.903 | 65.948 | 6 |
| unimts | 68.61 | 1nn | 32 | True | False | 67.315 | 67.435 | 6 |
| unimts | 68.61 | 1nn | 64 | True | False | 69.044 | 68.757 | 6 |
| unimts | 68.61 | 1nn | 128 | True | False | 70.975 | 71.041 | 6 |
| unimts | 68.61 | native_zero_support | 0 | True | False | 35.869 | 29.321 | 6 |
| unimts | 68.61 | prototype | 1 | True | False | 50.944 | 50.609 | 6 |
| unimts | 68.61 | prototype | 2 | True | False | 54.123 | 54.326 | 6 |
| unimts | 68.61 | prototype | 4 | True | False | 56.994 | 57.66 | 6 |
| unimts | 68.61 | prototype | 8 | True | False | 59.283 | 60.104 | 6 |
| unimts | 68.61 | prototype | 16 | True | False | 61.583 | 62.18 | 6 |
| unimts | 68.61 | prototype | 32 | True | False | 63.078 | 63.411 | 6 |
| unimts | 68.61 | prototype | 64 | True | False | 64.334 | 63.915 | 6 |
| unimts | 68.61 | prototype | 128 | True | False | 66.0 | 65.891 | 6 |
| unimts | 68.61 | ridge | 1 | True | False | 48.613 | 47.969 | 6 |
| unimts | 68.61 | ridge | 2 | True | False | 52.315 | 51.858 | 6 |
| unimts | 68.61 | ridge | 4 | True | False | 56.013 | 55.979 | 6 |
| unimts | 68.61 | ridge | 8 | True | False | 59.116 | 59.257 | 6 |
| unimts | 68.61 | ridge | 16 | True | False | 62.807 | 62.729 | 6 |
| unimts | 68.61 | ridge | 32 | True | False | 65.931 | 65.682 | 6 |
| unimts | 68.61 | ridge | 64 | True | False | 68.964 | 68.214 | 6 |
| unimts | 68.61 | ridge | 128 | True | False | 71.456 | 71.389 | 6 |

## 8-second windows

### inclusivehar
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | stream | n_devices | accuracy | f1_macro | balanced_accuracy | n_queries | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halo | 2.203 | halo-classifier | 0 | True | True | phone_waist | 1 | 40.26717557251908 | 36.62548771717269 | 40.12915921940254 | 1048 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_waist | 1 | 38.54961832061068 | 34.845634308673674 | 38.35226681694301 | 1048 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_waist | 1 | 33.20610687022901 | 32.66954581026534 | 33.308611452326375 | 1048 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_waist | 1 | 33.20610687022901 | 32.66954581026534 | 33.308611452326375 | 1048 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_waist | 1 | 32.729007633587784 | 32.26305805297431 | 32.821989375531 | 1048 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_waist | 1 | 36.54580152671756 | 35.16194513398579 | 36.59569728021675 | 1048 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_waist | 1 | 33.01526717557252 | 32.15513364460197 | 33.10021524567494 | 1048 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_waist | 1 | 33.68320610687023 | 33.44638914051911 | 33.78684979753264 | 1048 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_waist | 1 | 35.20992366412214 | 34.60527900215678 | 35.36100452902014 | 1048 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_waist | 1 | 34.541984732824424 | 34.157536443871955 | 34.68886053808254 | 1048 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_waist | 1 | 38.645038167938935 | 36.65165102316303 | 38.65973525144888 | 1048 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_waist | 1 | 33.49236641221374 | 33.04848429794395 | 33.60446249456186 | 1048 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_waist | 1 | 36.45038167938932 | 36.416127291655485 | 36.590022963451844 | 1048 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_waist | 1 | 38.45419847328244 | 37.26996037573526 | 38.620193726560025 | 1048 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_waist | 1 | 37.97709923664122 | 37.358756262873456 | 38.15755323341414 | 1048 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_waist | 1 | 39.31297709923664 | 37.1125499400011 | 39.30044844314675 | 1048 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_waist | 1 | 37.5 | 37.179887771180745 | 37.6098239763704 | 1048 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_waist | 1 | 36.73664122137404 | 37.03314657408117 | 37.01072021851957 | 1048 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_waist | 1 | 39.12213740458015 | 37.72158295864207 | 39.260914848435995 | 1048 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_waist | 1 | 38.54961832061068 | 38.04582336795953 | 38.74420160361494 | 1048 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_waist | 1 | 40.0763358778626 | 37.558740831877095 | 40.059164647996475 | 1048 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_waist | 1 | 38.83587786259542 | 38.703943853357494 | 39.12215737792379 | 1048 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_waist | 1 | 37.11832061068702 | 37.40240817245027 | 37.33446249214954 | 1048 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_waist | 1 | 41.221374045801525 | 39.65144259876619 | 41.384066660589625 | 1048 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_waist | 1 | 41.412213740458014 | 40.61772794897283 | 41.61327661441785 | 1048 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_waist | 1 | 39.79007633587786 | 37.20672938771589 | 39.71384384839944 | 1048 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_waist | 1 | 39.31297709923664 | 39.17779143678865 | 39.5208838142327 | 1048 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_waist | 1 | 36.92748091603054 | 37.6110469983453 | 37.149834367521194 | 1048 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_waist | 1 | 42.17557251908397 | 40.414459989127856 | 42.4137624459762 | 1048 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_waist | 1 | 42.55725190839695 | 41.98334617290173 | 42.86814921147177 | 1048 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_waist | 1 | 39.408396946564885 | 36.82167804298857 | 39.35382570172721 | 1048 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_waist | 1 | 40.64885496183206 | 40.97623285293804 | 40.92340010332522 | 1048 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_waist | 1 | 37.11832061068702 | 38.06815374116053 | 37.365847319049514 | 1048 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_waist | 1 | 42.843511450381676 | 41.024258580853505 | 43.07905963352545 | 1048 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_waist | 1 | 41.79389312977099 | 40.970916353328214 | 42.15547671721286 | 1048 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_waist | 1 | 39.79007633587786 | 36.962405353952185 | 39.750100128368416 | 1048 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_waist | 1 | 39.88549618320611 | 40.289552070567545 | 40.13382326069966 | 1048 | ok |
| halo | 0.789 | 1nn | 128 | True | True | phone_waist | 1 | 36.06870229007634 | 37.292205793514185 | 36.36031736028096 | 1048 | ok |
| halo | 0.789 | prototype | 128 | True | True | phone_waist | 1 | 42.461832061068705 | 40.38663889653439 | 42.75876612393748 | 1048 | ok |
| halo | 0.789 | ridge | 128 | True | True | phone_waist | 1 | 42.55725190839695 | 41.488573942840986 | 42.98904796832362 | 1048 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | phone_waist | 1 | 39.217557251908396 | 36.5453413880685 | 39.180476436874386 | 1048 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | phone_waist | 1 | 40.26717557251908 | 40.90386806773409 | 40.57440016195071 | 1048 | ok |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_waist | 1 | 30.05725190839695 | 26.20716501522189 | 30.033517137857206 | 1048 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_waist | 1 | 28.530534351145036 | 28.58286787380772 | 28.67565652017073 | 1048 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_waist | 1 | 28.530534351145036 | 28.58286787380772 | 28.67565652017073 | 1048 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_waist | 1 | 27.671755725190838 | 27.626890467366355 | 27.775693403052898 | 1048 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_waist | 1 | 30.15267175572519 | 30.200971601336242 | 30.315782245456706 | 1048 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_waist | 1 | 29.580152671755727 | 29.62581274369192 | 29.748936291576626 | 1048 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_waist | 1 | 29.48473282442748 | 29.329771659314236 | 29.680730971654544 | 1048 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_waist | 1 | 30.438931297709924 | 30.643502152961066 | 30.48767703606558 | 1048 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_waist | 1 | 29.103053435114507 | 29.39287156466997 | 29.15563178887672 | 1048 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_waist | 1 | 30.248091603053435 | 30.22108683568397 | 30.33846572407855 | 1048 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_waist | 1 | 30.248091603053435 | 30.269861171888437 | 30.359944453681504 | 1048 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_waist | 1 | 29.00763358778626 | 29.210866249837526 | 29.09832413927573 | 1048 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_waist | 1 | 31.583969465648853 | 31.42384595819446 | 31.694338304248742 | 1048 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_waist | 1 | 29.00763358778626 | 29.169596781427252 | 29.137541219315562 | 1048 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_waist | 1 | 30.248091603053435 | 30.404203823555058 | 30.363711942647885 | 1048 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_waist | 1 | 32.729007633587784 | 32.43076425937942 | 32.984177729184275 | 1048 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_waist | 1 | 30.05725190839695 | 30.453948549355193 | 30.23560478058641 | 1048 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_waist | 1 | 31.297709923664126 | 31.255179671805937 | 31.468842246180177 | 1048 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_waist | 1 | 35.20992366412214 | 34.102664944175125 | 35.58363859482974 | 1048 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_waist | 1 | 30.725190839694655 | 31.096474466865587 | 30.92185573533664 | 1048 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_waist | 1 | 33.587786259541986 | 33.58400520273882 | 33.76405171369295 | 1048 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_waist | 1 | 38.645038167938935 | 37.2104219740185 | 39.062368372177694 | 1048 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | phone_waist | 1 | 29.00763358778626 | 29.482275198684654 | 29.196193083064774 | 1048 | ok |
| harnet | 4.49 | prototype | 128 | False | False | phone_waist | 1 | 29.580152671755727 | 29.24091120305131 | 29.71268571365752 | 1048 | ok |
| harnet | 4.49 | ridge | 128 | False | False | phone_waist | 1 | 37.11832061068702 | 35.72797285836393 | 37.58176152882484 | 1048 | ok |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_waist | 1 | 33.20610687022901 | 31.60466525851062 | 33.319779546849745 | 1048 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_waist | 1 | 32.156488549618324 | 32.1244772588768 | 32.262769907497244 | 1048 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_waist | 1 | 32.156488549618324 | 32.1244772588768 | 32.262769907497244 | 1048 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_waist | 1 | 29.866412213740457 | 29.819613740978486 | 30.07025853116094 | 1048 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_waist | 1 | 32.44274809160305 | 32.54405766725996 | 32.66395485988813 | 1048 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_waist | 1 | 31.583969465648853 | 31.53188374189116 | 31.811915522637292 | 1048 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_waist | 1 | 31.583969465648853 | 31.33733798396234 | 31.862353951563854 | 1048 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_waist | 1 | 36.06870229007634 | 36.15492810140361 | 36.25887225900388 | 1048 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_waist | 1 | 34.25572519083969 | 34.21404308318741 | 34.43484464911056 | 1048 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_waist | 1 | 35.973282442748086 | 35.426191925867336 | 36.30964940150445 | 1048 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_waist | 1 | 37.404580152671755 | 37.61805970915524 | 37.736409922926214 | 1048 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_waist | 1 | 33.587786259541986 | 33.513558561144464 | 33.813907552754216 | 1048 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_waist | 1 | 37.30916030534351 | 36.681391711687446 | 37.74735563361959 | 1048 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_waist | 1 | 38.645038167938935 | 39.03705118616396 | 38.92325584822533 | 1048 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_waist | 1 | 36.45038167938932 | 36.29209371419764 | 36.796931720804764 | 1048 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_waist | 1 | 41.603053435114504 | 40.56638225784901 | 42.114412084424806 | 1048 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_waist | 1 | 38.83587786259542 | 39.067467842969926 | 39.072361075817 | 1048 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_waist | 1 | 36.16412213740458 | 36.06937761473364 | 36.512387757934825 | 1048 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_waist | 1 | 42.36641221374045 | 41.34254457853818 | 42.8789855628021 | 1048 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_waist | 1 | 40.0763358778626 | 40.44411945887562 | 40.30224450662575 | 1048 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_waist | 1 | 39.217557251908396 | 38.90516950862043 | 39.60277008870695 | 1048 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_waist | 1 | 48.18702290076336 | 47.53772839175461 | 48.660696777771115 | 1048 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | phone_waist | 1 | 41.030534351145036 | 41.27926728301423 | 41.28937018283725 | 1048 | ok |
| unimts | 68.61 | prototype | 128 | True | False | phone_waist | 1 | 37.97709923664122 | 36.677729200721004 | 38.4311473355367 | 1048 | ok |
| unimts | 68.61 | ridge | 128 | True | False | phone_waist | 1 | 51.24045801526718 | 50.76668619743949 | 51.64114532360917 | 1048 | ok |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_waist | 1 | 16.889312977099237 | 4.816326530612245 | 16.666666666666664 | 1048 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_waist | 1 | 23.568702290076335 | 23.540573361146667 | 23.495520986762152 | 1048 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_waist | 1 | 23.568702290076335 | 23.540573361146667 | 23.495520986762152 | 1048 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_waist | 1 | 21.946564885496183 | 21.92522093691074 | 21.988836449086318 | 1048 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_waist | 1 | 20.706106870229007 | 20.74890810002811 | 20.716524255745075 | 1048 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_waist | 1 | 21.087786259541986 | 21.02488000612853 | 21.051447227161656 | 1048 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_waist | 1 | 21.660305343511453 | 21.370192326014656 | 21.67939138893551 | 1048 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_waist | 1 | 22.519083969465647 | 22.54555869637525 | 22.602891461155153 | 1048 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_waist | 1 | 24.141221374045802 | 24.16424167519624 | 24.265732846633405 | 1048 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_waist | 1 | 24.80916030534351 | 23.94674890723062 | 24.97051117734013 | 1048 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_waist | 1 | 23.377862595419845 | 23.366926772179646 | 23.42002409035159 | 1048 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_waist | 1 | 23.18702290076336 | 23.02383993146139 | 23.250210225415227 | 1048 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_waist | 1 | 24.52290076335878 | 23.35807660890083 | 24.628682692482865 | 1048 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_waist | 1 | 24.61832061068702 | 24.58120391415354 | 24.621191329752445 | 1048 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_waist | 1 | 23.66412213740458 | 23.237079120183015 | 23.852464872907756 | 1048 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_waist | 1 | 24.236641221374043 | 22.636860696972313 | 24.487454983151878 | 1048 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_waist | 1 | 23.568702290076335 | 23.639530078566203 | 23.610133461751357 | 1048 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_waist | 1 | 24.33206106870229 | 23.628891087725414 | 24.570656316479244 | 1048 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_waist | 1 | 25.0 | 22.153360030435866 | 25.291837382013483 | 1048 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_waist | 1 | 25.954198473282442 | 25.99387236944039 | 25.950242854663024 | 1048 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_waist | 1 | 24.045801526717558 | 22.61576127026984 | 24.364013877572923 | 1048 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_waist | 1 | 24.52290076335878 | 20.57736153024147 | 24.893117252387754 | 1048 | ok |
| normwear | 1293.86 | 1nn | 128 | True | False | phone_waist | 1 | 25.954198473282442 | 25.96594694984819 | 25.936660928637245 | 1048 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | phone_waist | 1 | 24.33206106870229 | 21.931136788062926 | 24.75519188013727 | 1048 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_waist | 1 | 31.297709923664126 | 27.971709888736047 | 31.50570762739873 | 1048 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_waist | 1 | 23.85496183206107 | 23.804979954831143 | 23.882224853412414 | 1048 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_waist | 1 | 23.85496183206107 | 23.804979954831143 | 23.882224853412414 | 1048 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_waist | 1 | 25.858778625954198 | 25.094238933881368 | 25.951962898944984 | 1048 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_waist | 1 | 24.80916030534351 | 24.85090780424445 | 24.90333932111829 | 1048 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_waist | 1 | 22.137404580152673 | 22.201290612565703 | 22.213475217909952 | 1048 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_waist | 1 | 23.85496183206107 | 22.864132225939514 | 23.960790738848416 | 1048 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_waist | 1 | 25.095419847328245 | 25.437576000740908 | 25.217712142954674 | 1048 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_waist | 1 | 23.66412213740458 | 23.571487570895265 | 23.760859655360143 | 1048 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_waist | 1 | 26.431297709923662 | 24.796632495357365 | 26.616190359559972 | 1048 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_waist | 1 | 27.003816793893133 | 26.957835664973747 | 27.231117581639047 | 1048 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_waist | 1 | 28.530534351145036 | 27.961347985845013 | 28.68710888251284 | 1048 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_waist | 1 | 32.25190839694657 | 29.350397747149387 | 32.568943559350146 | 1048 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_waist | 1 | 29.961832061068705 | 30.452154106603512 | 30.143001380237706 | 1048 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_waist | 1 | 29.293893129770993 | 27.303175620720204 | 29.520423697590232 | 1048 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_waist | 1 | 34.35114503816794 | 29.1933525773031 | 34.82676648512708 | 1048 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_waist | 1 | 27.29007633587786 | 27.869566509251868 | 27.50667372204446 | 1048 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_waist | 1 | 29.580152671755727 | 26.8826831285731 | 29.91449855208999 | 1048 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_waist | 1 | 36.06870229007634 | 28.90854618082887 | 36.694042323019154 | 1048 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_waist | 1 | 27.958015267175572 | 28.518944015959775 | 28.121490250370467 | 1048 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_waist | 1 | 29.48473282442748 | 25.63406525955207 | 29.90068762390841 | 1048 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_waist | 1 | 35.68702290076336 | 27.995353626644366 | 36.346128374614544 | 1048 | ok |
| limubert_x | 0.055 | 1nn | 128 | False | False | phone_waist | 1 | 26.14503816793893 | 27.09770591263712 | 26.336776799916873 | 1048 | ok |
| limubert_x | 0.055 | prototype | 128 | False | False | phone_waist | 1 | 27.86259541984733 | 22.737052861629405 | 28.378650646916075 | 1048 | ok |
| limubert_x | 0.055 | ridge | 128 | False | False | phone_waist | 1 | 35.400763358778626 | 27.457018854235198 | 36.087221395201944 | 1048 | ok |

### motionsense
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | stream | n_devices | accuracy | f1_macro | balanced_accuracy | n_queries | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halo | 2.203 | halo-classifier | 0 | True | True | phone_front_pocket | 1 | 77.57920389926889 | 69.93836299582844 | 73.1004337661814 | 3693 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_front_pocket | 1 | 71.13457893311671 | 62.88646846589233 | 68.09314698296998 | 3693 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_front_pocket | 1 | 74.81722177091795 | 71.0310398596186 | 72.20638715472475 | 3693 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_front_pocket | 1 | 74.81722177091795 | 71.0310398596186 | 72.20638715472475 | 3693 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_front_pocket | 1 | 74.84430002707826 | 70.83187008043726 | 72.25535895493903 | 3693 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_front_pocket | 1 | 79.52883834281073 | 74.62042414784077 | 76.23163968469578 | 3693 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_front_pocket | 1 | 74.11318711075006 | 70.27270736052108 | 71.6775816463492 | 3693 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_front_pocket | 1 | 79.25805578120769 | 75.60471126495183 | 76.70349284254448 | 3693 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_front_pocket | 1 | 80.63904684538315 | 76.76366161384288 | 77.68053928318285 | 3693 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_front_pocket | 1 | 81.0452206877877 | 76.65405386004007 | 77.76064405933454 | 3693 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_front_pocket | 1 | 82.42621175196318 | 78.13839901237274 | 79.40577024683971 | 3693 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_front_pocket | 1 | 79.85377741673436 | 75.93604068784056 | 77.09379025970779 | 3693 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_front_pocket | 1 | 83.5093419983753 | 80.5522132170549 | 81.9390624310282 | 3693 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_front_pocket | 1 | 85.9734633089629 | 82.72781648896141 | 83.57866236353274 | 3693 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_front_pocket | 1 | 86.94828053073383 | 83.42054053451616 | 84.3275750543364 | 3693 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_front_pocket | 1 | 85.0257243433523 | 80.8688707723908 | 81.77471787450413 | 3693 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_front_pocket | 1 | 84.72786352558896 | 81.42382817208359 | 82.63379230170925 | 3693 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_front_pocket | 1 | 86.73165448145139 | 83.96834475478245 | 84.90101696415003 | 3693 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_front_pocket | 1 | 88.11264554562686 | 85.26634887443923 | 86.0944544224258 | 3693 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_front_pocket | 1 | 89.49363660980232 | 86.72641424841883 | 87.44931322068888 | 3693 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_front_pocket | 1 | 87.46276739777959 | 84.14317688079453 | 84.7175316455597 | 3693 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_front_pocket | 1 | 88.32927159490929 | 85.63733003773277 | 86.41267561217131 | 3693 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_front_pocket | 1 | 88.00433252098566 | 85.20425687133097 | 85.96589631245905 | 3693 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_front_pocket | 1 | 89.00622799891687 | 86.31648787991978 | 86.93407514046419 | 3693 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_front_pocket | 1 | 91.33495802870296 | 89.0722661714243 | 89.49957553711377 | 3693 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_front_pocket | 1 | 88.22095857026807 | 85.1205916459229 | 85.3971919017474 | 3693 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_front_pocket | 1 | 89.71026265908475 | 86.95966317419561 | 87.4366046005093 | 3693 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_front_pocket | 1 | 91.33495802870296 | 89.07827936548469 | 89.53167754876378 | 3693 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_front_pocket | 1 | 89.76441917140536 | 87.21713487555375 | 87.73063439616419 | 3693 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_front_pocket | 1 | 92.33685350663417 | 90.34894300550441 | 90.7208589936948 | 3693 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_front_pocket | 1 | 88.38342810722989 | 85.15270329667715 | 85.45050448547565 | 3693 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_front_pocket | 1 | 92.36393176279446 | 90.14396512941612 | 90.33187035157475 | 3693 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_front_pocket | 1 | 92.85134037367993 | 90.87947749619408 | 91.10097455753673 | 3693 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_front_pocket | 1 | 90.14351475764961 | 87.70319989721801 | 88.20363330665757 | 3693 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_front_pocket | 1 | 93.09504467912267 | 91.21995263849878 | 91.58768946869652 | 3693 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_front_pocket | 1 | 88.95207148659627 | 85.73756452064599 | 85.8408027013415 | 3693 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_front_pocket | 1 | 93.25751421608447 | 91.26157303727656 | 91.20070107071618 | 3693 | ok |
| halo | 0.789 | 1nn | 128 | True | True | phone_front_pocket | 1 | 94.36772271865692 | 92.7438183279246 | 92.8941675064432 | 3693 | ok |
| halo | 0.789 | prototype | 128 | True | True | phone_front_pocket | 1 | 90.19767126997021 | 87.78384068307709 | 88.33211697882034 | 3693 | ok |
| halo | 0.789 | ridge | 128 | True | True | phone_front_pocket | 1 | 93.74492282696994 | 91.99738674162265 | 92.22532610015234 | 3693 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | phone_front_pocket | 1 | 89.2228540481993 | 86.17267730452122 | 86.24304506004648 | 3693 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | phone_front_pocket | 1 | 93.74492282696994 | 91.79814480920702 | 91.59354605660978 | 3693 | ok |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_front_pocket | 1 | 56.728946655835365 | 51.73408459958338 | 53.26043698367419 | 3693 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_front_pocket | 1 | 61.16978066612511 | 58.383770971486115 | 59.25617889030846 | 3693 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_front_pocket | 1 | 61.16978066612511 | 58.383770971486115 | 59.25617889030846 | 3693 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_front_pocket | 1 | 61.061467641483894 | 57.627925878631636 | 58.266141359419464 | 3693 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_front_pocket | 1 | 64.44624966152179 | 62.82418691404191 | 63.919491943104546 | 3693 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_front_pocket | 1 | 64.74411047928513 | 62.628580387688494 | 63.61093978616551 | 3693 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_front_pocket | 1 | 65.2315190901706 | 62.38731276071678 | 63.06107199032043 | 3693 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_front_pocket | 1 | 68.18304901164365 | 67.0697659747902 | 67.97814708100644 | 3693 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_front_pocket | 1 | 69.67235310046033 | 68.03738104223562 | 68.91362483212528 | 3693 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_front_pocket | 1 | 70.83671811535336 | 68.81908901407367 | 69.41361301874626 | 3693 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_front_pocket | 1 | 71.81153533712428 | 71.40568925037654 | 72.13790946963672 | 3693 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_front_pocket | 1 | 72.92174383969673 | 71.80355303018425 | 72.64019254832613 | 3693 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_front_pocket | 1 | 75.00676956404007 | 73.92826837754471 | 74.17171522430587 | 3693 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_front_pocket | 1 | 75.68372596804765 | 75.61702872441676 | 76.31954336610049 | 3693 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_front_pocket | 1 | 75.46709991876523 | 74.48229986813159 | 75.40732276694267 | 3693 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_front_pocket | 1 | 79.63715136745193 | 79.56483867954147 | 79.85503168679888 | 3693 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_front_pocket | 1 | 78.12076902247496 | 78.5016762046725 | 79.04655403623613 | 3693 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_front_pocket | 1 | 76.08989981045221 | 75.7963175280018 | 76.7084029028804 | 3693 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_front_pocket | 1 | 81.01814243162741 | 81.59367123834556 | 81.88616408468961 | 3693 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_front_pocket | 1 | 80.2057947468183 | 80.63585997898026 | 81.13154888127737 | 3693 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_front_pocket | 1 | 76.44191714053615 | 76.03523795254215 | 76.90495577994271 | 3693 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_front_pocket | 1 | 83.78012455997835 | 84.38678697726503 | 84.74425566612709 | 3693 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | phone_front_pocket | 1 | 82.23666395884105 | 82.77222277368983 | 83.08944420924922 | 3693 | ok |
| harnet | 4.49 | prototype | 128 | False | False | phone_front_pocket | 1 | 76.875169239101 | 76.63253918072277 | 77.49504981637548 | 3693 | ok |
| harnet | 4.49 | ridge | 128 | False | False | phone_front_pocket | 1 | 85.40481992959653 | 86.02756490608844 | 86.36711178676734 | 3693 | ok |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_front_pocket | 1 | 49.79691307879773 | 36.1472575328154 | 46.60347135616871 | 3693 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_front_pocket | 1 | 67.64148388843758 | 67.59684783062681 | 67.93239702763138 | 3693 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_front_pocket | 1 | 67.64148388843758 | 67.59684783062681 | 67.93239702763138 | 3693 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_front_pocket | 1 | 63.904684538315735 | 64.44558610094091 | 65.02499201091437 | 3693 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_front_pocket | 1 | 74.76306525859735 | 74.69823741162728 | 74.96846831762157 | 3693 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_front_pocket | 1 | 69.4828053073382 | 70.3260964901821 | 70.72135596697028 | 3693 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_front_pocket | 1 | 68.75169239101002 | 69.80930821782775 | 70.34181905886574 | 3693 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_front_pocket | 1 | 79.85377741673436 | 80.24197815651856 | 80.50719446173747 | 3693 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_front_pocket | 1 | 74.76306525859735 | 76.38153500893114 | 76.72184793622604 | 3693 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_front_pocket | 1 | 74.41104792851341 | 76.16545756267634 | 76.35417393334089 | 3693 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_front_pocket | 1 | 83.04901164365015 | 83.02535563017277 | 83.21382222749797 | 3693 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_front_pocket | 1 | 78.01245599783374 | 79.41830252855291 | 79.74544824301934 | 3693 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_front_pocket | 1 | 78.36447332791768 | 79.91039134218235 | 80.28200541531585 | 3693 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_front_pocket | 1 | 86.19008935824533 | 86.18681527996222 | 86.58698359174016 | 3693 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_front_pocket | 1 | 81.09937720010831 | 82.495221981568 | 83.09300900593789 | 3693 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_front_pocket | 1 | 82.50744652044408 | 83.95533674496697 | 84.4688612346942 | 3693 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_front_pocket | 1 | 88.89791497427566 | 88.8165385368234 | 88.98010459965556 | 3693 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_front_pocket | 1 | 80.85567289466559 | 82.31322749269107 | 82.78705873870392 | 3693 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_front_pocket | 1 | 82.96777687516924 | 84.51015549796223 | 84.90391569122423 | 3693 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_front_pocket | 1 | 89.16869753587869 | 89.25053500453458 | 89.4444907861411 | 3693 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_front_pocket | 1 | 81.34308150555104 | 82.80958606395835 | 83.42308668303751 | 3693 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_front_pocket | 1 | 84.13214189006229 | 85.60361410730293 | 85.99429401561972 | 3693 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | phone_front_pocket | 1 | 90.52261034389385 | 90.68207385032952 | 90.64330139270949 | 3693 | ok |
| unimts | 68.61 | prototype | 128 | True | False | phone_front_pocket | 1 | 81.34308150555104 | 82.99932073183443 | 83.66323602012233 | 3693 | ok |
| unimts | 68.61 | ridge | 128 | True | False | phone_front_pocket | 1 | 85.91930679664229 | 87.29858599413505 | 87.60606038636446 | 3693 | ok |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_front_pocket | 1 | 22.88112645545627 | 6.206845893932717 | 16.666666666666664 | 3693 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_front_pocket | 1 | 21.662604928242622 | 20.914664531729425 | 21.534189161251756 | 3693 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_front_pocket | 1 | 21.662604928242622 | 20.914664531729425 | 21.534189161251756 | 3693 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_front_pocket | 1 | 21.418900622799892 | 20.07523383787617 | 20.90792835953091 | 3693 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_front_pocket | 1 | 25.6431085838072 | 24.606952195047402 | 25.356459396146473 | 3693 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_front_pocket | 1 | 24.72244787435689 | 23.478057422980704 | 24.220231018943274 | 3693 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_front_pocket | 1 | 23.937178445708096 | 21.432747706308362 | 22.73061753065807 | 3693 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_front_pocket | 1 | 28.621716761440563 | 27.61835990716755 | 28.4249815314785 | 3693 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_front_pocket | 1 | 27.78229082047116 | 25.88716217747267 | 27.013364174453784 | 3693 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_front_pocket | 1 | 26.455456268616302 | 23.048420279470985 | 25.648253732391733 | 3693 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_front_pocket | 1 | 32.38559436772272 | 31.322760511570554 | 32.17494706246529 | 3693 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_front_pocket | 1 | 27.75521256431086 | 25.262868819141083 | 26.364813455845187 | 3693 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_front_pocket | 1 | 27.105334416463577 | 22.56634571335487 | 25.705549526996922 | 3693 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_front_pocket | 1 | 36.20362848632548 | 35.6167432413286 | 36.71699079548021 | 3693 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_front_pocket | 1 | 29.08204711616572 | 26.486340792949004 | 28.22586080659406 | 3693 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_front_pocket | 1 | 28.107229894394802 | 23.11580387237955 | 27.30603966345275 | 3693 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_front_pocket | 1 | 41.61927971838614 | 40.878732932471564 | 41.98727400185776 | 3693 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_front_pocket | 1 | 28.892499323043598 | 26.06656260734448 | 28.131284779766148 | 3693 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_front_pocket | 1 | 29.867316544814514 | 24.270515784770534 | 29.598294115356065 | 3693 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_front_pocket | 1 | 44.57080963985919 | 44.05037246236097 | 45.015102056991836 | 3693 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_front_pocket | 1 | 28.513403736799347 | 25.63454376192083 | 28.646431417838148 | 3693 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_front_pocket | 1 | 32.629298673165444 | 27.403517558081354 | 33.067332561876526 | 3693 | ok |
| normwear | 1293.86 | 1nn | 128 | True | False | phone_front_pocket | 1 | 49.038721906309235 | 48.53108443113647 | 49.585688959898384 | 3693 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | phone_front_pocket | 1 | 29.786081776333607 | 26.732218068737428 | 29.95760877246907 | 3693 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | phone_front_pocket | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_front_pocket | 1 | 50.17600866504197 | 43.13424485276043 | 49.809809306694916 | 3693 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_front_pocket | 1 | 55.51042512862172 | 53.29944315902885 | 54.52437866613532 | 3693 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_front_pocket | 1 | 55.51042512862172 | 53.29944315902885 | 54.52437866613532 | 3693 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_front_pocket | 1 | 51.854860546980774 | 47.670548689869086 | 52.624648969374924 | 3693 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_front_pocket | 1 | 67.12699702139182 | 64.60502232415077 | 65.82089129336302 | 3693 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_front_pocket | 1 | 64.60871919848363 | 62.19968653755258 | 63.77509542494369 | 3693 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_front_pocket | 1 | 58.51611156241538 | 53.959934138855814 | 59.4278167616307 | 3693 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_front_pocket | 1 | 74.60059572163553 | 71.84940984708247 | 72.90005355580216 | 3693 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_front_pocket | 1 | 67.64148388843758 | 65.45465261858384 | 66.73178106907622 | 3693 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_front_pocket | 1 | 60.03249390739236 | 56.39292348466638 | 62.33418198076956 | 3693 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_front_pocket | 1 | 81.50555104251286 | 78.79149991214217 | 79.26873548457499 | 3693 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_front_pocket | 1 | 72.05523964256702 | 70.1532023344038 | 71.50039108183435 | 3693 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_front_pocket | 1 | 65.2856756024912 | 61.800688806106685 | 68.33397607791669 | 3693 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_front_pocket | 1 | 85.51313295423775 | 83.38264567143335 | 83.70816730313314 | 3693 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_front_pocket | 1 | 74.03195234226916 | 72.7657377944448 | 73.78660223881276 | 3693 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_front_pocket | 1 | 69.04955320877335 | 65.4803605664823 | 73.30386360249267 | 3693 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_front_pocket | 1 | 87.81478472786353 | 85.94159570081302 | 86.36630883075078 | 3693 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_front_pocket | 1 | 75.2233956133225 | 74.09993689579166 | 75.32385596114437 | 3693 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_front_pocket | 1 | 72.43433522881126 | 68.56699995066195 | 77.22832375613224 | 3693 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_front_pocket | 1 | 90.089358245329 | 88.43429278225852 | 88.51350172607327 | 3693 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_front_pocket | 1 | 75.71080422420796 | 74.71969250301805 | 75.95125310786251 | 3693 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_front_pocket | 1 | 74.84430002707826 | 70.84800092282136 | 79.40755195898909 | 3693 | ok |
| limubert_x | 0.055 | 1nn | 128 | False | False | phone_front_pocket | 1 | 90.76631464933659 | 89.34431990086624 | 89.55142725152756 | 3693 | ok |
| limubert_x | 0.055 | prototype | 128 | False | False | phone_front_pocket | 1 | 76.14405632277281 | 75.17505018462397 | 76.26283139292434 | 3693 | ok |
| limubert_x | 0.055 | ridge | 128 | False | False | phone_front_pocket | 1 | 76.55023016517735 | 72.95485672650739 | 80.90508113215172 | 3693 | ok |

### realworld
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | stream | n_devices | accuracy | f1_macro | balanced_accuracy | n_queries | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halo | 2.203 | halo-classifier | 0 | True | True | phone_waist | 1 | 46.8384074941452 | 40.64419661621726 | 52.57004444620933 | 8540 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_waist | 1 | 29.578454332552695 | 20.900831640087635 | 35.200523078927695 | 8540 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_waist | 1 | 59.18032786885246 | 58.862794033088676 | 62.04596976921288 | 8540 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_waist | 1 | 59.18032786885246 | 58.862794033088676 | 62.04596976921288 | 8540 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_waist | 1 | 59.34426229508196 | 58.02498114203583 | 62.17298525414028 | 8540 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_waist | 1 | 60.17564402810305 | 58.736719518918164 | 63.4756045064641 | 8540 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_waist | 1 | 59.648711943793906 | 59.28359300065924 | 62.00130828078716 | 8540 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_waist | 1 | 65.85480093676816 | 65.93325315208689 | 68.59231657193709 | 8540 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_waist | 1 | 64.30913348946136 | 63.78907523588088 | 66.9980253312608 | 8540 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_waist | 1 | 65.59718969555036 | 64.33465770247544 | 68.34579977922675 | 8540 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_waist | 1 | 66.0304449648712 | 64.71906488500073 | 69.39228708167381 | 8540 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_waist | 1 | 67.04918032786885 | 67.25429481998941 | 69.56899912702038 | 8540 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_waist | 1 | 70.52693208430914 | 70.99926790951638 | 72.72467652563563 | 8540 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_waist | 1 | 70.85480093676814 | 70.97487959216222 | 73.34926576878482 | 8540 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_waist | 1 | 72.07259953161592 | 71.42537571851109 | 74.57790345906906 | 8540 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_waist | 1 | 69.85948477751757 | 68.6497625916557 | 72.83575697971114 | 8540 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_waist | 1 | 72.27166276346604 | 72.97106713169728 | 74.28085125193084 | 8540 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_waist | 1 | 72.24824355971897 | 72.9871571710136 | 74.66558768321583 | 8540 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_waist | 1 | 74.8360655737705 | 75.5008144661438 | 77.16796652391868 | 8540 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_waist | 1 | 76.44028103044496 | 76.26227625874749 | 78.37863384951781 | 8540 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_waist | 1 | 72.54098360655738 | 71.2891139278033 | 75.2058860907202 | 8540 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_waist | 1 | 75.84309133489462 | 76.87408351250514 | 77.82668810269332 | 8540 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_waist | 1 | 73.17330210772833 | 73.84170046694098 | 75.46078642595897 | 8540 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_waist | 1 | 77.42388758782201 | 78.16145528914413 | 79.19565400642834 | 8540 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_waist | 1 | 78.4543325526932 | 78.59864931683185 | 80.34466354980339 | 8540 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_waist | 1 | 73.59484777517564 | 72.22854493201613 | 76.2693778630118 | 8540 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_waist | 1 | 77.52927400468384 | 78.41485067861385 | 79.16946275769534 | 8540 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_waist | 1 | 73.79391100702576 | 74.7532609597465 | 76.07737592778275 | 8540 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_waist | 1 | 79.08665105386416 | 80.05721750874453 | 80.83145272678908 | 8540 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_waist | 1 | 79.5784543325527 | 79.66745857345524 | 81.33782166027792 | 8540 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_waist | 1 | 74.14519906323184 | 72.7072665917078 | 76.72448458794537 | 8540 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_waist | 1 | 78.55971896955504 | 79.28036865496576 | 80.3139390431222 | 8540 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_waist | 1 | 74.98829039812647 | 76.11417750742491 | 77.25936716480246 | 8540 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_waist | 1 | 79.20374707259953 | 79.97602231366321 | 80.81864929721858 | 8540 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_waist | 1 | 80.16393442622952 | 80.35402704670645 | 81.90180354321251 | 8540 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_waist | 1 | 74.36768149882904 | 73.10298046770474 | 76.96878580518317 | 8540 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_waist | 1 | 79.49648711943794 | 80.4757896753104 | 81.09745200360281 | 8540 | ok |
| halo | 0.789 | 1nn | 128 | True | True | phone_waist | 1 | 74.63700234192038 | 75.7047216432802 | 76.89085954963595 | 8540 | ok |
| halo | 0.789 | prototype | 128 | True | True | phone_waist | 1 | 79.54332552693208 | 80.33288477336693 | 81.10823758127435 | 8540 | ok |
| halo | 0.789 | ridge | 128 | True | True | phone_waist | 1 | 80.88992974238876 | 81.23138347818161 | 82.64172379379 | 8540 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | phone_waist | 1 | 74.75409836065575 | 73.53196679943112 | 77.31281327246904 | 8540 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | phone_waist | 1 | 79.5784543325527 | 80.57690744054406 | 81.17097112948377 | 8540 | ok |
| halo | 2.203 | halo-classifier | 0 | True | True | phone_forearm | 1 | 51.16101046185252 | 51.26348479423315 | 55.94165983620669 | 3919 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_forearm | 1 | 39.24470528195968 | 26.92198224745915 | 34.76710948930867 | 3919 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_forearm | 1 | 47.716254146465936 | 47.54536907454671 | 48.757861356120145 | 3919 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_forearm | 1 | 47.716254146465936 | 47.54536907454671 | 48.757861356120145 | 3919 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_forearm | 1 | 47.9203878540444 | 46.976797424123006 | 49.09542330749247 | 3919 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_forearm | 1 | 50.191375350854806 | 50.892228812899795 | 53.52266356185672 | 3919 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_forearm | 1 | 47.86935442714979 | 47.15143595010233 | 48.89467386385817 | 3919 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_forearm | 1 | 53.55958152589947 | 54.976260029139134 | 56.66478037319047 | 3919 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_forearm | 1 | 53.55958152589947 | 54.40475390054077 | 56.38431367785065 | 3919 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_forearm | 1 | 54.401633069660626 | 54.0049009099693 | 56.82304539895213 | 3919 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_forearm | 1 | 55.014034192396025 | 57.55573506824152 | 60.269591751400654 | 3919 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_forearm | 1 | 54.17198264863485 | 55.4907050450615 | 57.785081711283695 | 3919 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_forearm | 1 | 55.422301607552946 | 57.18261097704477 | 58.54378087370027 | 3919 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_forearm | 1 | 57.999489665731055 | 59.328780720036356 | 61.081242366127285 | 3919 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_forearm | 1 | 57.999489665731055 | 58.13628732166338 | 61.31622005081269 | 3919 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_forearm | 1 | 56.08573615718295 | 58.85927789958362 | 61.57718503550225 | 3919 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_forearm | 1 | 56.519520285787195 | 58.180034958857775 | 60.03057406538521 | 3919 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_forearm | 1 | 58.7905077825976 | 60.51002614913139 | 61.888163398321105 | 3919 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_forearm | 1 | 61.77596325593263 | 62.7209824395931 | 64.91901338721658 | 3919 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_forearm | 1 | 61.724929829038025 | 61.533511028700424 | 64.03051885775429 | 3919 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_forearm | 1 | 59.709109466700696 | 62.52720048896187 | 64.90525564885064 | 3919 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_forearm | 1 | 61.18907884664455 | 62.652004158908944 | 64.83873164963614 | 3919 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_forearm | 1 | 59.504975759122225 | 61.64288214721427 | 63.29560906110683 | 3919 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_forearm | 1 | 63.84281704516458 | 65.35837879145143 | 67.86183223300586 | 3919 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_forearm | 1 | 64.65935187547845 | 65.24290144672528 | 67.47512764597636 | 3919 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_forearm | 1 | 61.0104618525134 | 63.52050337087975 | 65.81466801882148 | 3919 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_forearm | 1 | 63.40903291656035 | 64.9518393681914 | 67.59001710108728 | 3919 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_forearm | 1 | 60.474610870119925 | 62.36768137020241 | 64.04129586366551 | 3919 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_forearm | 1 | 64.81245215616228 | 65.9509558822632 | 68.47332754627269 | 3919 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_forearm | 1 | 66.90482265884154 | 67.36128688381746 | 69.15713640833667 | 3919 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_forearm | 1 | 62.260780811431495 | 64.61781592377143 | 67.01019400104238 | 3919 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_forearm | 1 | 65.50140341923961 | 66.80296020413127 | 69.60021403194328 | 3919 | ok |
| halo | 0.789 | all | 64 | True | True | phone_forearm | 1 |  |  |  |  | n/a |
| halo | 0.789 | all | 128 | True | True | phone_forearm | 1 |  |  |  |  | n/a |
| halo | 2.203 | halo-classifier | 0 | True | True | phone_thigh | 1 | 44.48714159030446 | 41.052334678236505 | 44.28521808859912 | 3383 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_thigh | 1 | 21.489801950931124 | 15.730918494815308 | 25.152358008050218 | 3383 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_thigh | 1 | 49.334909843334316 | 47.126368229630415 | 51.048320765465036 | 3383 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_thigh | 1 | 49.334909843334316 | 47.126368229630415 | 51.048320765465036 | 3383 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_thigh | 1 | 49.80786284362991 | 47.31140310536709 | 51.84616178187633 | 3383 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_thigh | 1 | 47.53177652970736 | 45.61787545112949 | 50.379479244892664 | 3383 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_thigh | 1 | 49.364469405852795 | 46.98837187766267 | 50.85200651475401 | 3383 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_thigh | 1 | 57.700266036062665 | 55.926372586963794 | 59.16664349478021 | 3383 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_thigh | 1 | 55.95625184747266 | 53.89982453855936 | 58.24916883869802 | 3383 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_thigh | 1 | 58.58705291161691 | 56.18289633031532 | 60.699702416605405 | 3383 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_thigh | 1 | 53.94620159621638 | 52.834203411795656 | 56.612400684973885 | 3383 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_thigh | 1 | 58.97132722435708 | 57.201341553783266 | 60.50446758328445 | 3383 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_thigh | 1 | 61.2769731007981 | 59.90515588174215 | 63.36263696157198 | 3383 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_thigh | 1 | 59.8876736624298 | 57.94514280055767 | 62.661494924329276 | 3383 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_thigh | 1 | 63.34614247709134 | 60.72670641984057 | 65.17369407150292 | 3383 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_thigh | 1 | 56.961276973100794 | 56.19277414331185 | 60.01853852475707 | 3383 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_thigh | 1 | 63.493940289683714 | 61.504321841594425 | 65.273671784321 | 3383 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_thigh | 1 | 63.67129766479456 | 63.11766884345469 | 66.60023431666654 | 3383 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_thigh | 1 | 64.76500147797812 | 63.04399549414135 | 66.50972807209646 | 3383 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_thigh | 1 | 69.43541235589713 | 68.70627686061785 | 71.17006444215224 | 3383 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_thigh | 1 | 61.77948566361218 | 61.096942205734535 | 64.4441862750727 | 3383 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_thigh | 1 | 67.6027194797517 | 66.84946978558057 | 69.7548065722675 | 3383 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_thigh | 1 | 64.14425066509015 | 63.591647756484114 | 66.76633214664459 | 3383 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_thigh | 1 | 67.98699379249187 | 67.31179083797485 | 69.81714007675438 | 3383 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_thigh | 1 | 71.6819391073012 | 71.0357062057186 | 72.89318927847481 | 3383 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_thigh | 1 | 62.873189476795744 | 62.68425550265273 | 65.48363325187638 | 3383 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_thigh | 1 | 70.44043748152528 | 69.28597997423218 | 71.76566613704833 | 3383 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_thigh | 1 | 64.88323972805202 | 64.75664377881809 | 67.56942157740485 | 3383 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_thigh | 1 | 68.60774460537984 | 69.03226040468469 | 70.46677277533158 | 3383 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_thigh | 1 | 73.2190363582619 | 73.1925786718708 | 74.70135922532424 | 3383 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_thigh | 1 | 63.22790422701744 | 63.560813783325045 | 66.3338890533472 | 3383 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_thigh | 1 | 71.38634348211646 | 70.26627675844762 | 72.60347830092194 | 3383 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_thigh | 1 | 65.16516516516516 | 63.71624479800324 | 69.1738225962371 | 3330 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_thigh | 1 | 69.009009009009 | 68.54079156110966 | 70.3777994381348 | 3330 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_thigh | 1 | 73.48348348348348 | 72.50115871257151 | 75.28376195081742 | 3330 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_thigh | 1 | 62.58258258258258 | 60.37680152115462 | 66.70062827787224 | 3330 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_thigh | 1 | 71.83183183183182 | 70.11579985470601 | 74.16651163631772 | 3330 | ok |
| halo | 0.789 | all | 128 | True | True | phone_thigh | 1 |  |  |  |  | n/a |
| halo | 2.203 | halo-classifier | 0 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 57.446808510638306 | 63.11219230131577 | 67.35681701776338 | 2068 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| halo | 0.789 | 1nn | 1 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 64.94197292069632 | 62.586253740866916 | 66.59657470858045 | 2068 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 64.94197292069632 | 62.586253740866916 | 66.59657470858045 | 2068 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 65.76402321083172 | 62.94783343473116 | 67.32484447054054 | 2068 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 59.62282398452611 | 60.2439052206612 | 67.23662731596333 | 2068 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 64.65183752417795 | 62.25662355050549 | 66.24285337006094 | 2068 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 71.37330754352031 | 70.24345096511793 | 72.20817599449916 | 2068 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 66.29593810444874 | 62.34755209453906 | 66.7653517056355 | 2068 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 69.72920696324951 | 67.03024451075063 | 70.44934372449052 | 2068 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 64.07156673114119 | 67.32127123782838 | 71.54116377802694 | 2068 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 71.80851063829788 | 70.59445923244381 | 72.81305924129803 | 2068 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 73.98452611218569 | 73.35114611091164 | 74.98256017270005 | 2068 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 73.54932301740811 | 71.1784456224188 | 74.10800028689124 | 2068 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 75.77369439071566 | 73.70274834195234 | 76.10520532808465 | 2068 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 66.29593810444874 | 70.39615375563211 | 73.38796961644006 | 2068 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 76.64410058027079 | 75.75758688199075 | 77.27517330746736 | 2068 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 75.14506769825918 | 74.48481526996122 | 75.43825669138484 | 2068 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 75.67698259187621 | 74.29307382649091 | 76.32171558247674 | 2068 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 78.57833655705996 | 77.42164573403969 | 78.5337038189635 | 2068 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 68.3752417794971 | 72.64897645619865 | 74.46970683379786 | 2068 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 78.4816247582205 | 77.9206449213627 | 78.81429101408786 | 2068 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 74.41972920696325 | 72.39212617316437 | 74.38706666235504 | 2068 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 77.12765957446808 | 75.58350064503006 | 77.05569820648593 | 2068 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 80.12572533849129 | 78.85643383406573 | 79.74397756731781 | 2068 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 68.27852998065764 | 72.13110164294598 | 74.20743172602596 | 2068 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 77.8046421663443 | 76.54949094709731 | 78.07083005091627 | 2068 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 74.66732380482996 | 61.312079294015966 | 71.26172341418369 | 2029 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 75.75160177427304 | 71.83948261286383 | 73.43545954208686 | 2029 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 80.92656481025135 | 77.87712586322472 | 78.84841814401209 | 2029 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 67.66880236569739 | 60.682011911282416 | 71.23097563716495 | 2029 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 78.01872843765402 | 65.52537773711819 | 75.92385581693642 | 2029 | ok |
| halo | 0.789 | all | 64 | True | True | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| halo | 0.789 | all | 128 | True | True | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_waist | 1 | 35.78454332552693 | 28.669084996158933 | 38.07895977240254 | 8540 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_waist | 1 | 41.87353629976581 | 39.207824934076875 | 40.55157951234513 | 8540 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_waist | 1 | 41.87353629976581 | 39.207824934076875 | 40.55157951234513 | 8540 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_waist | 1 | 42.01405152224824 | 39.008521867393696 | 40.399656882752986 | 8540 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_waist | 1 | 45.93676814988291 | 43.26422027373165 | 44.62936573327653 | 8540 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_waist | 1 | 46.557377049180324 | 43.997031489871866 | 45.482323680403084 | 8540 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_waist | 1 | 47.306791569086656 | 44.21647115621275 | 45.89909844607484 | 8540 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_waist | 1 | 49.62529274004684 | 47.57642545079453 | 49.777494660429475 | 8540 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_waist | 1 | 50.948477751756435 | 48.83009749791514 | 51.267615956730936 | 8540 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_waist | 1 | 52.40046838407494 | 50.06753909311767 | 52.92974832924799 | 8540 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_waist | 1 | 53.36065573770492 | 51.302418918050904 | 53.223163304522856 | 8540 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_waist | 1 | 55.85480093676814 | 53.585566560930296 | 55.80620980399231 | 8540 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_waist | 1 | 58.53629976580797 | 56.07489156523936 | 58.45729510802031 | 8540 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_waist | 1 | 56.75644028103044 | 55.15832706669237 | 57.135314237930324 | 8540 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_waist | 1 | 58.65339578454333 | 56.92420942296247 | 59.30278337052788 | 8540 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_waist | 1 | 63.05620608899297 | 61.6307788776586 | 63.7693468079097 | 8540 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_waist | 1 | 59.21545667447307 | 57.876622086484616 | 59.40832158763614 | 8540 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_waist | 1 | 60.62060889929742 | 58.78643547952747 | 60.83587946341884 | 8540 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_waist | 1 | 67.03747072599532 | 66.6379964418337 | 68.14078016911859 | 8540 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_waist | 1 | 61.065573770491795 | 59.95006798475173 | 61.4773712360387 | 8540 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_waist | 1 | 61.8384074941452 | 59.84492324724935 | 61.66915026022117 | 8540 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_waist | 1 | 69.5433255269321 | 69.67692969265687 | 70.34850681221468 | 8540 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | phone_waist | 1 | 62.34192037470726 | 61.86036187752855 | 63.3327619688415 | 8540 | ok |
| harnet | 4.49 | prototype | 128 | False | False | phone_waist | 1 | 62.06088992974239 | 60.54604214941766 | 62.38599497748149 | 8540 | ok |
| harnet | 4.49 | ridge | 128 | False | False | phone_waist | 1 | 71.85011709601874 | 72.3273702433365 | 72.75431724975444 | 8540 | ok |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_forearm | 1 | 36.66751722378157 | 26.961321614848586 | 35.35000721823538 | 3919 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_forearm | 1 | 38.402653738198524 | 34.7663699050528 | 37.17795732280818 | 3919 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_forearm | 1 | 38.402653738198524 | 34.7663699050528 | 37.17795732280818 | 3919 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_forearm | 1 | 38.964021434039296 | 34.96458886848575 | 37.40346639068248 | 3919 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_forearm | 1 | 43.22531257973973 | 39.57258462747029 | 42.488439617187204 | 3919 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_forearm | 1 | 42.000510334268945 | 37.99821692491181 | 40.606346640470676 | 3919 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_forearm | 1 | 42.84256187803011 | 38.826240623527966 | 42.58398196284373 | 3919 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_forearm | 1 | 46.3383516203113 | 42.86569041305587 | 45.510835822292066 | 3919 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_forearm | 1 | 46.84868588925746 | 42.742230489146046 | 46.03804898025038 | 3919 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_forearm | 1 | 48.226588415412095 | 43.97677850688372 | 47.85082950984329 | 3919 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_forearm | 1 | 46.77213574891554 | 44.18101023793261 | 47.32265506991875 | 3919 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_forearm | 1 | 49.96172492982904 | 45.73689912073825 | 48.98324738634165 | 3919 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_forearm | 1 | 51.1099770349579 | 47.819847386792695 | 51.790191615227464 | 3919 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_forearm | 1 | 50.26792549119673 | 47.849729573445316 | 51.010510432913236 | 3919 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_forearm | 1 | 52.794080122480224 | 48.994159636664854 | 52.88150734735299 | 3919 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_forearm | 1 | 55.95815258994642 | 53.56134940117596 | 57.54097315504379 | 3919 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_forearm | 1 | 50.57412605256443 | 48.74225415142563 | 52.024333350581344 | 3919 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_forearm | 1 | 53.763715233477924 | 49.733660554848036 | 54.14671650974145 | 3919 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_forearm | 1 | 58.43327379433529 | 56.95560864194831 | 60.560569638450126 | 3919 | ok |
| harnet | 4.49 | all | 64 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| harnet | 4.49 | all | 128 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_thigh | 1 | 34.496009459060005 | 29.1681194894003 | 41.37407902970961 | 3383 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_thigh | 1 | 40.348802837718 | 39.024651696783465 | 42.70818469745955 | 3383 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_thigh | 1 | 40.348802837718 | 39.024651696783465 | 42.70818469745955 | 3383 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_thigh | 1 | 40.73307715045817 | 39.81953188610786 | 43.87465528232128 | 3383 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_thigh | 1 | 43.275199527047 | 42.99175409529767 | 47.625305714151345 | 3383 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_thigh | 1 | 44.102867277564286 | 43.7516124757096 | 48.46143814217366 | 3383 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_thigh | 1 | 45.7286432160804 | 45.968380805671906 | 51.02428681446058 | 3383 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_thigh | 1 | 45.10789240319244 | 45.54329555636728 | 49.52067338868805 | 3383 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_thigh | 1 | 47.79781259237363 | 48.26563595157876 | 52.853152520604276 | 3383 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_thigh | 1 | 49.394028968371266 | 49.87016961383205 | 55.43026805060076 | 3383 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_thigh | 1 | 49.86698196866686 | 49.931336477407164 | 53.258639469244194 | 3383 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_thigh | 1 | 50.54685190659178 | 51.46421486608993 | 56.03338571072778 | 3383 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_thigh | 1 | 54.41915459651197 | 55.759981516680114 | 60.839466634043845 | 3383 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_thigh | 1 | 50.36949453148093 | 51.27888989574922 | 54.65200691357231 | 3383 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_thigh | 1 | 53.23677209577298 | 54.77719475910649 | 59.4579614335151 | 3383 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_thigh | 1 | 56.81347916050843 | 58.703594716935356 | 63.75734282317074 | 3383 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_thigh | 1 | 53.029855158143654 | 53.649663664592616 | 56.994089451385456 | 3383 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_thigh | 1 | 54.21223765888264 | 55.27295795471673 | 60.45479714951125 | 3383 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_thigh | 1 | 58.291457286432156 | 60.25831324498885 | 64.90785954914031 | 3383 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_thigh | 1 | 55.07507507507508 | 53.276929640722194 | 59.68420505927947 | 3330 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_thigh | 1 | 55.31531531531532 | 54.21541218661395 | 63.32128535579166 | 3330 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_thigh | 1 | 59.669669669669666 | 59.28587043002294 | 66.55669435646918 | 3330 | ok |
| harnet | 4.49 | all | 128 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| harnet | 4.49 | zero_support | 0 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| harnet | 4.49 | 1nn | 1 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 50.29013539651837 | 46.59643506257234 | 50.68724652154313 | 2068 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 50.29013539651837 | 46.59643506257234 | 50.68724652154313 | 2068 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 49.90328820116054 | 46.17520477204583 | 51.07323264037803 | 2068 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 53.38491295938105 | 51.59333052077697 | 55.086082664541 | 2068 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 50.870406189555126 | 47.246673891992984 | 51.003858557032174 | 2068 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 50.870406189555126 | 47.54949128612008 | 51.84235804667446 | 2068 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 56.57640232108317 | 55.66707981342842 | 58.023879222103666 | 2068 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 55.12572533849129 | 52.770032144142256 | 57.82506014869027 | 2068 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 57.108317214700186 | 55.44354575670757 | 60.44284738563438 | 2068 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 59.477756286266924 | 58.33577794061233 | 59.9331202292483 | 2068 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 56.334622823984525 | 54.06433364503505 | 59.122293561632674 | 2068 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 60.299806576402325 | 60.503714637129356 | 64.67515623341565 | 2068 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 63.15280464216635 | 63.14851843090767 | 65.18050011312445 | 2068 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 59.864603481624755 | 58.364982893123354 | 63.65114484747707 | 2068 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 64.11992263056094 | 65.65265845235717 | 68.8754972617663 | 2068 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 64.9581074420897 | 56.64400106622664 | 65.97925943703214 | 2029 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 60.42385411532775 | 52.36137628851589 | 61.866672455971475 | 2029 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 65.69738787580089 | 56.76982946729321 | 67.28829177171008 | 2029 | ok |
| harnet | 4.49 | all | 64 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| harnet | 4.49 | all | 128 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_waist | 1 | 42.82201405152225 | 34.0024105789719 | 45.9452445724382 | 8540 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_waist | 1 | 58.524590163934434 | 58.05796626948574 | 60.493654038648245 | 8540 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_waist | 1 | 58.524590163934434 | 58.05796626948574 | 60.493654038648245 | 8540 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_waist | 1 | 57.18969555035128 | 55.93743452918307 | 59.36818582451945 | 8540 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_waist | 1 | 64.69555035128806 | 65.10003777829463 | 67.05835760127783 | 8540 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_waist | 1 | 62.31850117096018 | 62.58695429569448 | 64.8578031403623 | 8540 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_waist | 1 | 61.69789227166277 | 61.02503653108424 | 64.39644030246818 | 8540 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_waist | 1 | 68.27868852459017 | 69.27139227452479 | 70.39652472658015 | 8540 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_waist | 1 | 66.0304449648712 | 67.33789371725274 | 68.8109993163355 | 8540 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_waist | 1 | 65.49180327868852 | 65.74906277840688 | 68.23839894070086 | 8540 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_waist | 1 | 70.18735362997658 | 71.31180522541902 | 72.46675461965005 | 8540 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_waist | 1 | 69.9648711943794 | 71.66482979945032 | 72.38019467577087 | 8540 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_waist | 1 | 70.74941451990632 | 71.00369724635867 | 73.17557076084024 | 8540 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_waist | 1 | 72.52927400468384 | 73.48592227628889 | 74.61722125136929 | 8540 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_waist | 1 | 71.82669789227167 | 73.4083629269696 | 73.90537762246225 | 8540 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_waist | 1 | 73.25526932084308 | 72.71851142113795 | 75.4076000013265 | 8540 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_waist | 1 | 73.20843091334895 | 74.45676778676044 | 75.17701828905976 | 8540 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_waist | 1 | 73.59484777517564 | 75.35110004081169 | 75.56872078347357 | 8540 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_waist | 1 | 74.4496487119438 | 74.5363890411466 | 76.71270391936254 | 8540 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_waist | 1 | 74.29742388758783 | 75.98522048370016 | 76.39368842459946 | 8540 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_waist | 1 | 73.99297423887587 | 75.81066319096672 | 75.77629288844625 | 8540 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_waist | 1 | 75.7728337236534 | 76.02980492897775 | 77.97244707969838 | 8540 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | phone_waist | 1 | 75.74941451990632 | 77.17307591229297 | 77.86039991327476 | 8540 | ok |
| unimts | 68.61 | prototype | 128 | True | False | phone_waist | 1 | 74.8360655737705 | 76.53453659029164 | 76.60530338257787 | 8540 | ok |
| unimts | 68.61 | ridge | 128 | True | False | phone_waist | 1 | 76.99063231850117 | 77.5895395782713 | 79.315827582184 | 8540 | ok |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_forearm | 1 | 47.38453687165094 | 34.589981288315094 | 46.70451925496556 | 3919 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_forearm | 1 | 45.700433784128606 | 43.67941396310839 | 49.370451993016204 | 3919 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_forearm | 1 | 45.700433784128606 | 43.67941396310839 | 49.370451993016204 | 3919 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_forearm | 1 | 43.557029854554735 | 41.9021854681789 | 48.156127863834875 | 3919 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_forearm | 1 | 50.57412605256443 | 49.65019876067693 | 55.384076341566434 | 3919 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_forearm | 1 | 48.787956111252875 | 47.82005917341 | 53.50979072762655 | 3919 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_forearm | 1 | 47.25695330441439 | 45.930395353816664 | 52.32732738926293 | 3919 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_forearm | 1 | 54.68231691758102 | 53.37188962463533 | 58.549214593445775 | 3919 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_forearm | 1 | 52.692013268691 | 52.38111955944862 | 56.688214470997124 | 3919 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_forearm | 1 | 52.0796121459556 | 51.29892283374738 | 56.31511007704601 | 3919 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_forearm | 1 | 56.5705537126818 | 56.4344755678507 | 60.73916066241727 | 3919 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_forearm | 1 | 54.47818321000255 | 54.12076212751962 | 57.723276912795725 | 3919 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_forearm | 1 | 54.70783363102832 | 54.187644920783505 | 58.61279353846967 | 3919 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_forearm | 1 | 56.672620566471046 | 57.01920183114977 | 61.922233940161 | 3919 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_forearm | 1 | 56.8767542740495 | 56.13966907827928 | 59.96608484681363 | 3919 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_forearm | 1 | 57.74432253125797 | 56.60941011993048 | 61.3665611084774 | 3919 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_forearm | 1 | 58.89257463638683 | 59.086556321902904 | 63.820297385529216 | 3919 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_forearm | 1 | 58.38224036744067 | 57.04246661274459 | 61.157452579396185 | 3919 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_forearm | 1 | 59.86220974738453 | 58.3700981014931 | 63.44091077948577 | 3919 | ok |
| unimts | 68.61 | all | 64 | True | False | phone_forearm | 1 |  |  |  |  | n/a |
| unimts | 68.61 | all | 128 | True | False | phone_forearm | 1 |  |  |  |  | n/a |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_thigh | 1 | 32.75199527047 | 22.89203236709103 | 34.43578151776144 | 3383 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_thigh | 1 | 50.517292344073304 | 52.27917579525765 | 56.80538068400532 | 3383 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_thigh | 1 | 50.517292344073304 | 52.27917579525765 | 56.80538068400532 | 3383 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_thigh | 1 | 47.76825302985516 | 49.45875226043628 | 55.5441267777564 | 3383 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_thigh | 1 | 59.178244161986406 | 61.928929120166984 | 65.29106748990078 | 3383 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_thigh | 1 | 52.49778303281112 | 55.506712245866794 | 60.009954822834864 | 3383 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_thigh | 1 | 51.75879396984925 | 54.44541822265684 | 59.72141029054604 | 3383 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_thigh | 1 | 62.3411173514632 | 65.6901949116926 | 68.35934122183285 | 3383 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_thigh | 1 | 55.21726278451079 | 60.02979123833031 | 63.3834654246409 | 3383 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_thigh | 1 | 55.86757315991724 | 60.150375729770836 | 64.60542541630595 | 3383 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_thigh | 1 | 66.7750517292344 | 70.24814589582951 | 72.28114835511153 | 3383 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_thigh | 1 | 58.14365947383979 | 64.45134735041698 | 66.08292422521127 | 3383 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_thigh | 1 | 59.50339934968962 | 65.08960429733577 | 67.52015806520704 | 3383 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_thigh | 1 | 66.8932899793083 | 70.25141832336165 | 72.07278596122886 | 3383 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_thigh | 1 | 59.798994974874375 | 66.38510951189703 | 67.6806302884778 | 3383 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_thigh | 1 | 61.513449600945904 | 67.84605591916846 | 69.45395211391411 | 3383 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_thigh | 1 | 67.9574342299734 | 71.75218755269756 | 73.63740118117241 | 3383 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_thigh | 1 | 60.47886491279929 | 67.71687045750295 | 68.7623010455546 | 3383 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_thigh | 1 | 62.19331953887083 | 68.28226208643905 | 70.59721562776193 | 3383 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_thigh | 1 | 67.89789789789789 | 72.09878229478099 | 73.78718491960679 | 3330 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_thigh | 1 | 61.98198198198198 | 69.49953071173972 | 70.46843094139261 | 3330 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_thigh | 1 | 63.033033033033036 | 68.58963435189372 | 72.10468584095298 | 3330 | ok |
| unimts | 68.61 | all | 128 | True | False | phone_thigh | 1 |  |  |  |  | n/a |
| unimts | 68.61 | zero_support | 0 | True | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| unimts | 68.61 | 1nn | 1 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 55.947775628626694 | 55.05136755871176 | 60.571093331787274 | 2068 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 55.947775628626694 | 55.05136755871176 | 60.571093331787274 | 2068 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 54.69052224371374 | 53.21125488462811 | 59.85853431065416 | 2068 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 62.379110251450676 | 62.662582714149686 | 66.2629378212555 | 2068 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 57.30174081237911 | 56.60939948224236 | 61.2996349525597 | 2068 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 56.76982591876209 | 55.76250000193426 | 61.14633971152046 | 2068 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 65.81237911025146 | 66.55712731591134 | 69.55651660523013 | 2068 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 58.36557059961315 | 61.0179511943716 | 63.73007145014849 | 2068 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 58.9458413926499 | 61.70467003711065 | 64.98820854529518 | 2068 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 68.56866537717602 | 69.64131983499063 | 71.91295968024104 | 2068 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 60.49323017408123 | 64.23307815512307 | 66.23176142196465 | 2068 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 61.79883945841392 | 66.19331354116589 | 68.2638274844826 | 2068 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 71.08317214700193 | 71.31944311282116 | 73.93221710719251 | 2068 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 60.58994197292069 | 64.39681039925264 | 66.14643911410951 | 2068 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 63.44294003868471 | 66.72507382041465 | 69.14148295710658 | 2068 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 69.54164613109907 | 59.535248390471594 | 69.40806093900062 | 2029 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 62.74026614095614 | 62.258168101605506 | 63.94590372186679 | 2029 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 65.50024642681123 | 56.537318823249436 | 66.53084036538758 | 2029 | ok |
| unimts | 68.61 | all | 64 | True | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| unimts | 68.61 | all | 128 | True | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_waist | 1 | 14.192037470725996 | 3.126140183312214 | 12.510212418300654 | 8540 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_waist | 1 | 19.71896955503513 | 18.787720900271538 | 19.124721253468262 | 8540 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_waist | 1 | 19.71896955503513 | 18.787720900271538 | 19.124721253468262 | 8540 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_waist | 1 | 19.320843091334893 | 18.00936151401303 | 18.542832551033758 | 8540 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_waist | 1 | 21.510538641686182 | 20.46059367963277 | 20.6888303304802 | 8540 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_waist | 1 | 21.00702576112412 | 19.970952905006072 | 20.514354618520393 | 8540 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_waist | 1 | 20.22248243559719 | 18.44499766765204 | 19.321352294930172 | 8540 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_waist | 1 | 23.407494145199063 | 22.317009201515138 | 23.03735839035378 | 8540 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_waist | 1 | 21.510538641686182 | 20.060214799343022 | 20.088449664990094 | 8540 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_waist | 1 | 20.843091334894616 | 18.274908353690215 | 19.358633135968546 | 8540 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_waist | 1 | 25.210772833723656 | 24.09939324486014 | 25.16409482616389 | 8540 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_waist | 1 | 22.37704918032787 | 20.827655480207106 | 21.416492065225505 | 8540 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_waist | 1 | 22.833723653395786 | 19.511483391540434 | 21.100266519578557 | 8540 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_waist | 1 | 28.0327868852459 | 26.706212040170506 | 27.429025497668547 | 8540 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_waist | 1 | 22.833723653395786 | 21.2448903006309 | 21.979969301934073 | 8540 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_waist | 1 | 22.903981264637004 | 19.42899079975701 | 21.34725761118341 | 8540 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_waist | 1 | 29.203747072599533 | 27.71238632426124 | 28.10456549731996 | 8540 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_waist | 1 | 23.313817330210775 | 21.49054394999602 | 22.416893027022557 | 8540 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_waist | 1 | 25.07025761124122 | 21.252265938769842 | 23.370297615939045 | 8540 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_waist | 1 | 31.346604215456676 | 29.843823504192784 | 30.970196955851915 | 8540 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_waist | 1 | 23.337236533957846 | 21.07122054657272 | 22.027683917603692 | 8540 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_waist | 1 | 28.23185011709602 | 24.121142389959033 | 26.054488845353596 | 8540 | ok |
| normwear | 1293.86 | 1nn | 128 | True | False | phone_waist | 1 | 33.77049180327869 | 32.251137654917436 | 33.69108153851047 | 8540 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | phone_waist | 1 | 23.501170960187352 | 21.146969135242195 | 22.50138175298069 | 8540 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | phone_waist | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_forearm | 1 | 16.407246746619037 | 3.523673827268742 | 12.5 | 3919 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_forearm | 1 | 21.587139576422558 | 19.382092355448126 | 20.603359565782664 | 3919 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_forearm | 1 | 21.587139576422558 | 19.382092355448126 | 20.603359565782664 | 3919 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_forearm | 1 | 18.34651696861444 | 16.979054899699097 | 18.28271702659227 | 3919 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_forearm | 1 | 22.556774687420262 | 20.307653232638913 | 22.057105114312616 | 3919 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_forearm | 1 | 23.092625669813728 | 20.902883065848382 | 22.93178223874757 | 3919 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_forearm | 1 | 19.64786935442715 | 17.83325196087079 | 19.711963076188564 | 3919 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_forearm | 1 | 24.240877774942586 | 21.74387230650743 | 22.9990253360707 | 3919 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_forearm | 1 | 23.041592242919112 | 20.977502102227085 | 22.953456361677084 | 3919 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_forearm | 1 | 21.612656289869864 | 19.190191045306122 | 22.28296511938867 | 3919 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_forearm | 1 | 25.97601428935953 | 23.298764333457772 | 25.156584694969037 | 3919 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_forearm | 1 | 23.44985965807604 | 21.437082046413742 | 23.951936382972793 | 3919 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_forearm | 1 | 23.500893084970656 | 20.70926595015011 | 24.4936201102224 | 3919 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_forearm | 1 | 28.068384792038785 | 25.248510856356177 | 26.780057357442654 | 3919 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_forearm | 1 | 24.087777494258738 | 21.947076198053157 | 25.515447411430927 | 3919 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_forearm | 1 | 26.07808114314876 | 21.63669615160139 | 26.845184117770117 | 3919 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_forearm | 1 | 29.573870885429958 | 26.339235674632466 | 27.557817841070946 | 3919 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_forearm | 1 | 24.368461342179128 | 21.925026845666306 | 26.2406371258384 | 3919 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_forearm | 1 | 27.7877009441184 | 21.713493845869294 | 28.25955916575885 | 3919 | ok |
| normwear | 1293.86 | all | 64 | True | False | phone_forearm | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | all | 128 | True | False | phone_forearm | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_thigh | 1 | 19.065917824416196 | 4.0062111801242235 | 12.5 | 3383 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_thigh | 1 | 19.361513449600945 | 18.585622253548983 | 21.1255581045068 | 3383 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_thigh | 1 | 19.361513449600945 | 18.585622253548983 | 21.1255581045068 | 3383 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_thigh | 1 | 18.14957138634348 | 16.899060315546212 | 18.97967542684957 | 3383 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_thigh | 1 | 23.58853088974283 | 22.23227034935053 | 24.544419886940254 | 3383 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_thigh | 1 | 22.25835057641147 | 21.121374505075813 | 24.001655861667377 | 3383 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_thigh | 1 | 18.563405261602128 | 17.237518751046647 | 20.68144064659351 | 3383 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_thigh | 1 | 25.48034289092521 | 23.780505837798877 | 26.16390075580971 | 3383 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_thigh | 1 | 22.613065326633166 | 21.055590634098493 | 23.44123630690363 | 3383 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_thigh | 1 | 20.366538575229086 | 18.3196864350644 | 21.90957512269416 | 3383 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_thigh | 1 | 29.589122080993203 | 27.64913622976194 | 30.306912059506185 | 3383 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_thigh | 1 | 23.58853088974283 | 21.274953324500682 | 22.77721932525069 | 3383 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_thigh | 1 | 21.046408513154006 | 18.371457456788768 | 22.781270158957675 | 3383 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_thigh | 1 | 31.924327519952705 | 29.742464649750005 | 32.52209230730829 | 3383 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_thigh | 1 | 25.716819391073013 | 23.410713082026753 | 26.005742054877462 | 3383 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_thigh | 1 | 24.238841265149276 | 20.861183196087758 | 27.57968882387449 | 3383 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_thigh | 1 | 34.555128584096956 | 32.10631548024791 | 35.31641358073402 | 3383 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_thigh | 1 | 24.977830328111146 | 22.899365903116863 | 26.20141919991392 | 3383 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_thigh | 1 | 24.800472953000295 | 20.451943795305223 | 29.680466806523174 | 3383 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_thigh | 1 | 36.786786786786784 | 33.22075313637612 | 40.071558975425276 | 3330 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_thigh | 1 | 25.255255255255253 | 21.935217631640096 | 28.953422439331643 | 3330 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_thigh | 1 | 26.396396396396398 | 20.82953279365965 | 34.592241187625675 | 3330 | ok |
| normwear | 1293.86 | all | 128 | True | False | phone_thigh | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | zero_support | 0 | True | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 21.27659574468085 | 18.594940277649712 | 19.999773735776028 | 2068 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 21.27659574468085 | 18.594940277649712 | 19.999773735776028 | 2068 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 19.148936170212767 | 16.930136129602573 | 17.727861108863618 | 2068 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 23.93617021276596 | 21.174911410938265 | 23.37662389405531 | 2068 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 22.58220502901354 | 20.122162097208758 | 22.356200948501275 | 2068 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 20.6963249516441 | 17.79120774913892 | 19.434251256243368 | 2068 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 28.1431334622824 | 24.554827859220456 | 26.44062592425291 | 2068 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 24.032882011605416 | 21.004455645725606 | 22.828399845666553 | 2068 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 22.872340425531913 | 19.564678454811077 | 21.473129242160866 | 2068 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 32.05996131528047 | 29.070237187483887 | 32.4373904583574 | 2068 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 23.500967117988395 | 20.548150545651968 | 22.598595013149637 | 2068 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 25.0 | 20.895683409065175 | 23.660825801119344 | 2068 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 35.05802707930368 | 31.73940762184926 | 35.927126773425236 | 2068 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 25.338491295938105 | 22.02567215660528 | 23.61660713451108 | 2068 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 27.36943907156673 | 22.100863196710126 | 25.156704735008674 | 2068 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 36.71759487432232 | 30.777904563777792 | 35.75192229848857 | 2029 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 25.135534746180383 | 20.94883698424857 | 25.641972738622037 | 2029 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 31.591917200591425 | 23.473004040851883 | 30.689393195877557 | 2029 | ok |
| normwear | 1293.86 | all | 64 | True | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| normwear | 1293.86 | all | 128 | True | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 1 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 2 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 4 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 8 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 16 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 32 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 64 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 128 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 1 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 2 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 4 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 8 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 16 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 32 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 64 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 128 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 1 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 2 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 4 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 8 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 16 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 32 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 64 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 128 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | zero_support | 0 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 1 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 2 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 4 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 8 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 16 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 32 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 64 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 128 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |

### shoaib
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | stream | n_devices | accuracy | f1_macro | balanced_accuracy | n_queries | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halo | 2.203 | halo-classifier | 0 | True | True | phone_right_pocket | 1 | 70.1863354037267 | 68.11446348656096 | 70.18633540372672 | 1610 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_right_pocket | 1 | 58.38509316770186 | 55.28544919548297 | 58.385093167701875 | 1610 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_right_pocket | 1 | 75.3416149068323 | 75.84423815416991 | 75.34161490683229 | 1610 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_right_pocket | 1 | 75.3416149068323 | 75.84423815416991 | 75.34161490683229 | 1610 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_right_pocket | 1 | 77.01863354037268 | 77.06289670734265 | 77.01863354037268 | 1610 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_right_pocket | 1 | 79.25465838509317 | 79.3853542516884 | 79.25465838509317 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_right_pocket | 1 | 75.77639751552795 | 76.05002898806104 | 75.77639751552796 | 1610 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_right_pocket | 1 | 80.4968944099379 | 80.71761374406479 | 80.4968944099379 | 1610 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_right_pocket | 1 | 80.55900621118013 | 80.79939555995809 | 80.55900621118013 | 1610 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_right_pocket | 1 | 82.6086956521739 | 82.39143128681903 | 82.6086956521739 | 1610 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_right_pocket | 1 | 81.30434782608695 | 80.99296015967668 | 81.30434782608695 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_right_pocket | 1 | 81.05590062111801 | 81.14236023484371 | 81.05590062111801 | 1610 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_right_pocket | 1 | 85.34161490683229 | 85.37936603487866 | 85.3416149068323 | 1610 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_right_pocket | 1 | 86.08695652173914 | 86.12649629389398 | 86.08695652173914 | 1610 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_right_pocket | 1 | 87.51552795031056 | 87.35857992196699 | 87.51552795031054 | 1610 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_right_pocket | 1 | 84.34782608695653 | 84.07461437037439 | 84.34782608695653 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_right_pocket | 1 | 85.83850931677019 | 85.76993231621995 | 85.8385093167702 | 1610 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_right_pocket | 1 | 88.88198757763975 | 88.80038901599875 | 88.88198757763976 | 1610 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_right_pocket | 1 | 89.13043478260869 | 89.14890871404262 | 89.1304347826087 | 1610 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_right_pocket | 1 | 91.05590062111801 | 90.96611526561847 | 91.05590062111801 | 1610 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_right_pocket | 1 | 86.45962732919254 | 86.35847721203469 | 86.45962732919256 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_right_pocket | 1 | 89.44099378881988 | 89.34698495962434 | 89.44099378881988 | 1610 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_right_pocket | 1 | 90.43478260869566 | 90.3931954434435 | 90.43478260869566 | 1610 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_right_pocket | 1 | 90.24844720496894 | 90.26983558262526 | 90.24844720496894 | 1610 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_right_pocket | 1 | 93.1055900621118 | 93.066947956773 | 93.10559006211182 | 1610 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_right_pocket | 1 | 88.13664596273291 | 88.0035556015918 | 88.13664596273293 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_right_pocket | 1 | 90.0 | 89.91636607833732 | 90.0 | 1610 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_right_pocket | 1 | 90.8695652173913 | 90.83111851410403 | 90.86956521739131 | 1610 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_right_pocket | 1 | 90.06211180124224 | 90.12853271282401 | 90.06211180124222 | 1610 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_right_pocket | 1 | 94.65838509316771 | 94.63721229790562 | 94.65838509316772 | 1610 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_right_pocket | 1 | 88.32298136645963 | 88.2156392596948 | 88.32298136645963 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_right_pocket | 1 | 90.8695652173913 | 90.80022221092185 | 90.8695652173913 | 1610 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_right_pocket | 1 | 90.6832298136646 | 90.62484694546966 | 90.6832298136646 | 1610 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_right_pocket | 1 | 89.81366459627328 | 89.90331245637707 | 89.81366459627328 | 1610 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_right_pocket | 1 | 95.71428571428572 | 95.68497005597374 | 95.71428571428572 | 1610 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_right_pocket | 1 | 89.06832298136645 | 89.04372662749834 | 89.06832298136645 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_right_pocket | 1 | 90.6832298136646 | 90.5691758519803 | 90.6832298136646 | 1610 | ok |
| halo | 0.789 | 1nn | 128 | True | True | phone_right_pocket | 1 | 91.24223602484473 | 91.18548048361566 | 91.24223602484471 | 1610 | ok |
| halo | 0.789 | prototype | 128 | True | True | phone_right_pocket | 1 | 89.87577639751552 | 90.00972818486915 | 89.87577639751552 | 1610 | ok |
| halo | 0.789 | ridge | 128 | True | True | phone_right_pocket | 1 | 95.96273291925466 | 95.90986126443939 | 95.96273291925466 | 1610 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | phone_right_pocket | 1 | 89.13043478260869 | 89.08713280046277 | 89.1304347826087 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | phone_right_pocket | 1 | 90.8695652173913 | 90.74482116161646 | 90.8695652173913 | 1610 | ok |
| halo | 2.203 | halo-classifier | 0 | True | True | phone_left_pocket | 1 | 75.09316770186335 | 73.59885502503032 | 75.09316770186336 | 1610 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_left_pocket | 1 | 61.118012422360245 | 59.19751337235601 | 61.118012422360245 | 1610 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_left_pocket | 1 | 77.08074534161491 | 77.29724637233693 | 77.0807453416149 | 1610 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_left_pocket | 1 | 77.08074534161491 | 77.29724637233693 | 77.0807453416149 | 1610 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_left_pocket | 1 | 79.25465838509317 | 79.14975294769944 | 79.25465838509315 | 1610 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_left_pocket | 1 | 81.36645962732919 | 80.91080672003528 | 81.3664596273292 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_left_pocket | 1 | 77.08074534161491 | 77.07686040845002 | 77.08074534161489 | 1610 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_left_pocket | 1 | 80.62111801242236 | 80.62504360571852 | 80.62111801242237 | 1610 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_left_pocket | 1 | 83.29192546583852 | 83.21568825234247 | 83.29192546583852 | 1610 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_left_pocket | 1 | 85.15527950310559 | 84.91461687735581 | 85.15527950310559 | 1610 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_left_pocket | 1 | 84.16149068322981 | 83.56808024938323 | 84.16149068322981 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_left_pocket | 1 | 82.2360248447205 | 82.12700967519885 | 82.23602484472049 | 1610 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_left_pocket | 1 | 85.65217391304348 | 85.67679687209284 | 85.65217391304348 | 1610 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_left_pocket | 1 | 87.51552795031056 | 87.53582945818833 | 87.51552795031054 | 1610 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_left_pocket | 1 | 89.19254658385093 | 89.03185775623831 | 89.19254658385093 | 1610 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_left_pocket | 1 | 84.6583850931677 | 84.11095358497194 | 84.65838509316768 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_left_pocket | 1 | 86.0248447204969 | 85.84229116568062 | 86.0248447204969 | 1610 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_left_pocket | 1 | 88.88198757763975 | 88.84161105772387 | 88.88198757763975 | 1610 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_left_pocket | 1 | 88.26086956521739 | 88.25526813226026 | 88.2608695652174 | 1610 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_left_pocket | 1 | 90.6832298136646 | 90.59349006393418 | 90.6832298136646 | 1610 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_left_pocket | 1 | 86.8944099378882 | 86.60812251439232 | 86.89440993788821 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_left_pocket | 1 | 89.00621118012423 | 88.88790263880078 | 89.00621118012423 | 1610 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_left_pocket | 1 | 90.62111801242236 | 90.63471941043609 | 90.62111801242237 | 1610 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_left_pocket | 1 | 89.75155279503106 | 89.80328853324148 | 89.75155279503106 | 1610 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_left_pocket | 1 | 92.54658385093167 | 92.50450732715187 | 92.54658385093168 | 1610 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_left_pocket | 1 | 88.38509316770187 | 88.08364901500077 | 88.38509316770187 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_left_pocket | 1 | 90.6832298136646 | 90.5771672229811 | 90.6832298136646 | 1610 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_left_pocket | 1 | 92.54658385093167 | 92.52570130229886 | 92.54658385093168 | 1610 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_left_pocket | 1 | 90.12422360248448 | 90.13039531912578 | 90.12422360248448 | 1610 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_left_pocket | 1 | 93.7888198757764 | 93.77472723040506 | 93.7888198757764 | 1610 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_left_pocket | 1 | 88.94409937888199 | 88.7455329050601 | 88.94409937888197 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_left_pocket | 1 | 91.42857142857143 | 91.3265266165787 | 91.42857142857144 | 1610 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_left_pocket | 1 | 92.54658385093167 | 92.52456467219481 | 92.54658385093168 | 1610 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_left_pocket | 1 | 90.31055900621118 | 90.2979729925683 | 90.3105590062112 | 1610 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_left_pocket | 1 | 94.59627329192547 | 94.59267769352373 | 94.59627329192546 | 1610 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_left_pocket | 1 | 88.38509316770187 | 88.13350762819815 | 88.38509316770188 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_left_pocket | 1 | 91.42857142857143 | 91.34955272564098 | 91.42857142857143 | 1610 | ok |
| halo | 0.789 | 1nn | 128 | True | True | phone_left_pocket | 1 | 93.16770186335404 | 93.12627929211726 | 93.16770186335404 | 1610 | ok |
| halo | 0.789 | prototype | 128 | True | True | phone_left_pocket | 1 | 90.43478260869566 | 90.4357425013275 | 90.43478260869566 | 1610 | ok |
| halo | 0.789 | ridge | 128 | True | True | phone_left_pocket | 1 | 95.96273291925466 | 95.95802899338224 | 95.96273291925466 | 1610 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | phone_left_pocket | 1 | 88.81987577639751 | 88.61350075916633 | 88.81987577639751 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | phone_left_pocket | 1 | 91.5527950310559 | 91.47910663237325 | 91.55279503105591 | 1610 | ok |
| halo | 2.203 | halo-classifier | 0 | True | True | phone_belt | 1 | 75.15527950310559 | 74.72541873321042 | 75.1552795031056 | 1610 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_belt | 1 | 57.51552795031056 | 54.1133838960264 | 57.51552795031056 | 1610 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_belt | 1 | 84.03726708074534 | 84.04682188427286 | 84.03726708074534 | 1610 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_belt | 1 | 84.03726708074534 | 84.04682188427286 | 84.03726708074534 | 1610 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_belt | 1 | 84.472049689441 | 84.25568457348038 | 84.472049689441 | 1610 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_belt | 1 | 88.50931677018633 | 88.53217205714134 | 88.50931677018635 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_belt | 1 | 83.16770186335404 | 82.9619375141441 | 83.16770186335403 | 1610 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_belt | 1 | 89.62732919254658 | 89.67974399479182 | 89.6273291925466 | 1610 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_belt | 1 | 91.36645962732919 | 91.37434849475116 | 91.36645962732919 | 1610 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_belt | 1 | 90.99378881987577 | 90.88966231276953 | 90.99378881987577 | 1610 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_belt | 1 | 91.05590062111801 | 91.11225240606524 | 91.05590062111803 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_belt | 1 | 89.37888198757764 | 89.29150672997908 | 89.37888198757766 | 1610 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_belt | 1 | 91.92546583850931 | 91.95606474618194 | 91.9254658385093 | 1610 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_belt | 1 | 93.97515527950311 | 93.99982822200502 | 93.97515527950311 | 1610 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_belt | 1 | 93.22981366459628 | 93.2015150020722 | 93.22981366459628 | 1610 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_belt | 1 | 92.98136645962732 | 93.02677191882307 | 92.98136645962734 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_belt | 1 | 93.22981366459628 | 93.20086647871963 | 93.22981366459628 | 1610 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_belt | 1 | 93.72670807453416 | 93.74718437572096 | 93.72670807453416 | 1610 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_belt | 1 | 94.16149068322981 | 94.17799818292283 | 94.16149068322981 | 1610 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_belt | 1 | 94.40993788819875 | 94.41643782169426 | 94.40993788819877 | 1610 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_belt | 1 | 92.9192546583851 | 92.97251139169036 | 92.9192546583851 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_belt | 1 | 93.97515527950311 | 93.96109540832079 | 93.97515527950311 | 1610 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_belt | 1 | 95.40372670807453 | 95.41909376598652 | 95.40372670807454 | 1610 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_belt | 1 | 94.72049689440993 | 94.7339589783408 | 94.72049689440993 | 1610 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_belt | 1 | 95.27950310559005 | 95.27659434504912 | 95.27950310559005 | 1610 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_belt | 1 | 93.66459627329192 | 93.70054604809881 | 93.66459627329192 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_belt | 1 | 95.77639751552796 | 95.76054443192848 | 95.77639751552796 | 1610 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_belt | 1 | 94.59627329192547 | 94.60970016692622 | 94.59627329192547 | 1610 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_belt | 1 | 95.83850931677019 | 95.84648683010258 | 95.83850931677019 | 1610 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_belt | 1 | 96.14906832298136 | 96.15660493399835 | 96.14906832298136 | 1610 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_belt | 1 | 93.72670807453416 | 93.771145097664 | 93.72670807453416 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_belt | 1 | 95.77639751552796 | 95.76783260380861 | 95.77639751552795 | 1610 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_belt | 1 | 95.03105590062113 | 95.04350282361621 | 95.03105590062113 | 1610 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_belt | 1 | 95.77639751552796 | 95.78007300468123 | 95.77639751552795 | 1610 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_belt | 1 | 96.58385093167702 | 96.59270500803152 | 96.58385093167702 | 1610 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_belt | 1 | 93.72670807453416 | 93.75894483764199 | 93.72670807453416 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_belt | 1 | 96.45962732919254 | 96.45592006612125 | 96.45962732919256 | 1610 | ok |
| halo | 0.789 | 1nn | 128 | True | True | phone_belt | 1 | 94.84472049689441 | 94.87821891229949 | 94.8447204968944 | 1610 | ok |
| halo | 0.789 | prototype | 128 | True | True | phone_belt | 1 | 96.08695652173913 | 96.09553221003686 | 96.08695652173913 | 1610 | ok |
| halo | 0.789 | ridge | 128 | True | True | phone_belt | 1 | 97.14285714285714 | 97.14772502076381 | 97.14285714285715 | 1610 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | phone_belt | 1 | 93.66459627329192 | 93.69715078262433 | 93.66459627329192 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | phone_belt | 1 | 96.52173913043478 | 96.51557760264821 | 96.52173913043477 | 1610 | ok |
| halo | 2.203 | halo-classifier | 0 | True | True | watch_wrist_proxy | 1 | 61.118012422360245 | 58.51204621569368 | 61.11801242236024 | 1610 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | watch_wrist_proxy | 1 | 43.41614906832298 | 36.73157013017397 | 43.41614906832297 | 1610 | ok |
| halo | 0.789 | 1nn | 1 | True | True | watch_wrist_proxy | 1 | 79.3167701863354 | 79.23340718919266 | 79.31677018633542 | 1610 | ok |
| halo | 0.789 | prototype | 1 | True | True | watch_wrist_proxy | 1 | 79.3167701863354 | 79.23340718919266 | 79.31677018633542 | 1610 | ok |
| halo | 0.789 | ridge | 1 | True | True | watch_wrist_proxy | 1 | 80.24844720496894 | 80.09958169823904 | 80.24844720496894 | 1610 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | watch_wrist_proxy | 1 | 81.80124223602483 | 81.60491335979602 | 81.80124223602485 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | watch_wrist_proxy | 1 | 78.32298136645963 | 78.23386326293816 | 78.32298136645963 | 1610 | ok |
| halo | 0.789 | 1nn | 2 | True | True | watch_wrist_proxy | 1 | 82.17391304347827 | 82.14410719818652 | 82.17391304347827 | 1610 | ok |
| halo | 0.789 | prototype | 2 | True | True | watch_wrist_proxy | 1 | 84.34782608695653 | 84.24904599835843 | 84.34782608695653 | 1610 | ok |
| halo | 0.789 | ridge | 2 | True | True | watch_wrist_proxy | 1 | 84.16149068322981 | 83.80467477768796 | 84.16149068322981 | 1610 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | watch_wrist_proxy | 1 | 83.7888198757764 | 83.66763054237569 | 83.78881987577638 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | watch_wrist_proxy | 1 | 82.4223602484472 | 82.40176616080873 | 82.42236024844722 | 1610 | ok |
| halo | 0.789 | 1nn | 4 | True | True | watch_wrist_proxy | 1 | 84.16149068322981 | 84.15952750767008 | 84.16149068322981 | 1610 | ok |
| halo | 0.789 | prototype | 4 | True | True | watch_wrist_proxy | 1 | 87.20496894409938 | 87.11450114886576 | 87.20496894409938 | 1610 | ok |
| halo | 0.789 | ridge | 4 | True | True | watch_wrist_proxy | 1 | 87.01863354037268 | 86.61243885797256 | 87.01863354037268 | 1610 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | watch_wrist_proxy | 1 | 84.16149068322981 | 84.20362762664668 | 84.1614906832298 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | watch_wrist_proxy | 1 | 84.90683229813665 | 84.76898997377303 | 84.90683229813666 | 1610 | ok |
| halo | 0.789 | 1nn | 8 | True | True | watch_wrist_proxy | 1 | 84.72049689440993 | 84.49916703681791 | 84.72049689440995 | 1610 | ok |
| halo | 0.789 | prototype | 8 | True | True | watch_wrist_proxy | 1 | 87.5776397515528 | 87.39668181623183 | 87.5776397515528 | 1610 | ok |
| halo | 0.789 | ridge | 8 | True | True | watch_wrist_proxy | 1 | 87.51552795031056 | 87.13633428989066 | 87.51552795031056 | 1610 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | watch_wrist_proxy | 1 | 86.52173913043478 | 86.39314459323059 | 86.52173913043478 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | watch_wrist_proxy | 1 | 87.01863354037268 | 86.79357035941575 | 87.01863354037268 | 1610 | ok |
| halo | 0.789 | 1nn | 16 | True | True | watch_wrist_proxy | 1 | 85.77639751552795 | 85.86741398040581 | 85.77639751552793 | 1610 | ok |
| halo | 0.789 | prototype | 16 | True | True | watch_wrist_proxy | 1 | 89.25465838509317 | 89.06212176219267 | 89.25465838509315 | 1610 | ok |
| halo | 0.789 | ridge | 16 | True | True | watch_wrist_proxy | 1 | 90.24844720496894 | 90.00767417426341 | 90.24844720496894 | 1610 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | watch_wrist_proxy | 1 | 86.58385093167702 | 86.69792484387295 | 86.58385093167702 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | watch_wrist_proxy | 1 | 88.07453416149067 | 88.0526997680403 | 88.07453416149069 | 1610 | ok |
| halo | 0.789 | 1nn | 32 | True | True | watch_wrist_proxy | 1 | 85.71428571428571 | 85.71038232697975 | 85.71428571428571 | 1610 | ok |
| halo | 0.789 | prototype | 32 | True | True | watch_wrist_proxy | 1 | 89.93788819875776 | 89.80362293242408 | 89.93788819875776 | 1610 | ok |
| halo | 0.789 | ridge | 32 | True | True | watch_wrist_proxy | 1 | 90.43478260869566 | 90.26717995139089 | 90.43478260869566 | 1610 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | watch_wrist_proxy | 1 | 86.8944099378882 | 86.93188641559607 | 86.89440993788821 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | watch_wrist_proxy | 1 | 88.50931677018633 | 88.44728538281544 | 88.50931677018636 | 1610 | ok |
| halo | 0.789 | 1nn | 64 | True | True | watch_wrist_proxy | 1 | 86.83229813664596 | 86.86848006549582 | 86.83229813664595 | 1610 | ok |
| halo | 0.789 | prototype | 64 | True | True | watch_wrist_proxy | 1 | 89.75155279503106 | 89.61005658713746 | 89.75155279503106 | 1610 | ok |
| halo | 0.789 | ridge | 64 | True | True | watch_wrist_proxy | 1 | 90.74534161490682 | 90.60353561291478 | 90.74534161490682 | 1610 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | watch_wrist_proxy | 1 | 87.14285714285714 | 87.16211984943605 | 87.14285714285715 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | watch_wrist_proxy | 1 | 89.44099378881988 | 89.38221652965301 | 89.44099378881988 | 1610 | ok |
| halo | 0.789 | 1nn | 128 | True | True | watch_wrist_proxy | 1 | 86.58385093167702 | 86.61904221405169 | 86.58385093167702 | 1610 | ok |
| halo | 0.789 | prototype | 128 | True | True | watch_wrist_proxy | 1 | 89.93788819875776 | 89.80518421404152 | 89.93788819875776 | 1610 | ok |
| halo | 0.789 | ridge | 128 | True | True | watch_wrist_proxy | 1 | 91.36645962732919 | 91.28637528479932 | 91.3664596273292 | 1610 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | watch_wrist_proxy | 1 | 87.14285714285714 | 87.17721951083205 | 87.14285714285714 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | watch_wrist_proxy | 1 | 89.87577639751552 | 89.84515310449093 | 89.87577639751552 | 1610 | ok |
| halo | 2.203 | halo-classifier | 0 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 75.59006211180123 | 72.38074874422622 | 75.59006211180123 | 1610 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| halo | 0.789 | 1nn | 1 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 88.01242236024845 | 88.02321938094074 | 88.01242236024844 | 1610 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 88.01242236024845 | 88.02321938094074 | 88.01242236024844 | 1610 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 88.88198757763975 | 88.734269719725 | 88.88198757763975 | 1610 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 89.93788819875776 | 89.96842928971168 | 89.93788819875776 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 87.51552795031056 | 87.46443529740249 | 87.51552795031056 | 1610 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 90.8695652173913 | 90.83400636779496 | 90.8695652173913 | 1610 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 91.24223602484473 | 91.17542614884935 | 91.24223602484473 | 1610 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 92.73291925465838 | 92.56215696535219 | 92.7329192546584 | 1610 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 90.6832298136646 | 90.65739070195644 | 90.6832298136646 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 91.24223602484473 | 91.1985272229116 | 91.24223602484473 | 1610 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 93.41614906832298 | 93.39843008236409 | 93.41614906832298 | 1610 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 93.66459627329192 | 93.65712370796996 | 93.66459627329195 | 1610 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 94.78260869565217 | 94.7210430803033 | 94.78260869565217 | 1610 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 91.49068322981367 | 91.47512087615172 | 91.49068322981367 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 94.72049689440993 | 94.70092960323821 | 94.72049689440993 | 1610 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 95.46583850931677 | 95.45008162943839 | 95.46583850931677 | 1610 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 94.472049689441 | 94.46413573911549 | 94.472049689441 | 1610 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 96.08695652173913 | 96.07105441108791 | 96.08695652173914 | 1610 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 93.47826086956522 | 93.52424467159383 | 93.47826086956523 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 95.27950310559005 | 95.2651712610006 | 95.27950310559007 | 1610 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 96.83229813664596 | 96.80812111281155 | 96.83229813664596 | 1610 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 94.90683229813665 | 94.88960309369878 | 94.90683229813666 | 1610 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 96.95652173913044 | 96.92988328826996 | 96.95652173913044 | 1610 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 94.40993788819875 | 94.42306222115035 | 94.40993788819875 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 96.52173913043478 | 96.49806073701076 | 96.5217391304348 | 1610 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 97.0807453416149 | 97.07666719621623 | 97.0807453416149 | 1610 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 95.40372670807453 | 95.38193421121652 | 95.40372670807453 | 1610 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 97.70186335403727 | 97.69301377983098 | 97.70186335403727 | 1610 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 94.472049689441 | 94.49519808504878 | 94.472049689441 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 96.52173913043478 | 96.50137580263325 | 96.52173913043478 | 1610 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 97.14285714285714 | 97.13476107538169 | 97.14285714285715 | 1610 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 95.71428571428572 | 95.699385601941 | 95.71428571428572 | 1610 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 98.13664596273291 | 98.13043697917131 | 98.13664596273293 | 1610 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 94.40993788819875 | 94.42893711208636 | 94.40993788819875 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 96.8944099378882 | 96.88728811128226 | 96.8944099378882 | 1610 | ok |
| halo | 0.789 | 1nn | 128 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 96.95652173913044 | 96.93869729296922 | 96.95652173913044 | 1610 | ok |
| halo | 0.789 | prototype | 128 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 95.527950310559 | 95.50683333625211 | 95.527950310559 | 1610 | ok |
| halo | 0.789 | ridge | 128 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 98.75776397515527 | 98.75593105036813 | 98.75776397515527 | 1610 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 94.34782608695652 | 94.37667694355102 | 94.34782608695652 | 1610 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 97.14285714285714 | 97.12746495777826 | 97.14285714285714 | 1610 | ok |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_right_pocket | 1 | 44.03726708074534 | 40.321557638269276 | 44.03726708074534 | 1610 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_right_pocket | 1 | 62.422360248447205 | 61.55635168525272 | 62.4223602484472 | 1610 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_right_pocket | 1 | 62.422360248447205 | 61.55635168525272 | 62.4223602484472 | 1610 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_right_pocket | 1 | 63.22981366459627 | 62.27598149023989 | 63.22981366459627 | 1610 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_right_pocket | 1 | 67.32919254658385 | 66.80680929599605 | 67.32919254658385 | 1610 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_right_pocket | 1 | 66.77018633540372 | 66.09752086766227 | 66.77018633540372 | 1610 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_right_pocket | 1 | 65.96273291925466 | 65.06264413454971 | 65.96273291925465 | 1610 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_right_pocket | 1 | 73.47826086956522 | 73.23131089761107 | 73.47826086956522 | 1610 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_right_pocket | 1 | 72.17391304347827 | 71.50628942142801 | 72.17391304347827 | 1610 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_right_pocket | 1 | 72.60869565217392 | 71.60023783041255 | 72.6086956521739 | 1610 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_right_pocket | 1 | 77.20496894409938 | 76.93945994226141 | 77.20496894409938 | 1610 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_right_pocket | 1 | 76.64596273291926 | 76.06436001155991 | 76.64596273291926 | 1610 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_right_pocket | 1 | 79.00621118012423 | 78.2846799844579 | 79.00621118012423 | 1610 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_right_pocket | 1 | 81.49068322981367 | 81.33378967071661 | 81.49068322981367 | 1610 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_right_pocket | 1 | 77.63975155279503 | 77.10677636538192 | 77.63975155279505 | 1610 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_right_pocket | 1 | 82.04968944099379 | 81.5854673653502 | 82.04968944099379 | 1610 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_right_pocket | 1 | 83.35403726708076 | 83.21954843513927 | 83.35403726708074 | 1610 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_right_pocket | 1 | 77.32919254658384 | 76.86961019777975 | 77.32919254658384 | 1610 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_right_pocket | 1 | 83.35403726708076 | 83.03558695203085 | 83.35403726708074 | 1610 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_right_pocket | 1 | 87.51552795031056 | 87.47592833423784 | 87.51552795031057 | 1610 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_right_pocket | 1 | 77.08074534161491 | 76.54769579452699 | 77.08074534161491 | 1610 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_right_pocket | 1 | 86.83229813664596 | 86.58033384840141 | 86.83229813664596 | 1610 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | phone_right_pocket | 1 | 87.70186335403727 | 87.71838985202966 | 87.70186335403727 | 1610 | ok |
| harnet | 4.49 | prototype | 128 | False | False | phone_right_pocket | 1 | 77.14285714285715 | 76.66622944070721 | 77.14285714285715 | 1610 | ok |
| harnet | 4.49 | ridge | 128 | False | False | phone_right_pocket | 1 | 88.69565217391305 | 88.55142169970208 | 88.69565217391305 | 1610 | ok |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_left_pocket | 1 | 46.95652173913044 | 43.43718578521915 | 46.95652173913044 | 1610 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_left_pocket | 1 | 62.11180124223602 | 61.82317087904741 | 62.11180124223602 | 1610 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_left_pocket | 1 | 62.11180124223602 | 61.82317087904741 | 62.11180124223602 | 1610 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_left_pocket | 1 | 62.60869565217392 | 62.17678547014963 | 62.60869565217391 | 1610 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_left_pocket | 1 | 68.94409937888199 | 68.61874632938813 | 68.944099378882 | 1610 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_left_pocket | 1 | 68.75776397515529 | 68.46432582970884 | 68.75776397515529 | 1610 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_left_pocket | 1 | 69.19254658385093 | 68.70445470765596 | 69.19254658385093 | 1610 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_left_pocket | 1 | 74.09937888198758 | 73.68079951224183 | 74.09937888198759 | 1610 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_left_pocket | 1 | 73.66459627329192 | 73.24113373127177 | 73.66459627329192 | 1610 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_left_pocket | 1 | 75.15527950310559 | 74.60020035470485 | 75.15527950310558 | 1610 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_left_pocket | 1 | 79.8136645962733 | 79.59171961685068 | 79.81366459627328 | 1610 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_left_pocket | 1 | 75.46583850931677 | 75.068665195856 | 75.46583850931677 | 1610 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_left_pocket | 1 | 78.57142857142857 | 78.13880257299832 | 78.57142857142857 | 1610 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_left_pocket | 1 | 82.04968944099379 | 81.81973475824049 | 82.04968944099379 | 1610 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_left_pocket | 1 | 76.95652173913044 | 76.49904736741769 | 76.95652173913044 | 1610 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_left_pocket | 1 | 81.61490683229815 | 81.18458484724013 | 81.61490683229815 | 1610 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_left_pocket | 1 | 85.27950310559007 | 85.17929171879769 | 85.27950310559007 | 1610 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_left_pocket | 1 | 78.01242236024845 | 77.59939447197426 | 78.01242236024845 | 1610 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_left_pocket | 1 | 83.35403726708076 | 83.04266793767661 | 83.35403726708074 | 1610 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_left_pocket | 1 | 87.88819875776397 | 87.81003925371795 | 87.88819875776397 | 1610 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_left_pocket | 1 | 77.82608695652173 | 77.48345698655969 | 77.82608695652175 | 1610 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_left_pocket | 1 | 86.08695652173914 | 85.84320832279559 | 86.08695652173913 | 1610 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | phone_left_pocket | 1 | 89.3167701863354 | 89.29404193980997 | 89.3167701863354 | 1610 | ok |
| harnet | 4.49 | prototype | 128 | False | False | phone_left_pocket | 1 | 77.70186335403727 | 77.30063076091221 | 77.70186335403727 | 1610 | ok |
| harnet | 4.49 | ridge | 128 | False | False | phone_left_pocket | 1 | 88.57142857142857 | 88.34734194799651 | 88.57142857142856 | 1610 | ok |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_belt | 1 | 50.37267080745341 | 48.91031461373469 | 50.37267080745341 | 1610 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_belt | 1 | 58.94409937888199 | 57.91406635458268 | 58.94409937888199 | 1610 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_belt | 1 | 58.94409937888199 | 57.91406635458268 | 58.94409937888199 | 1610 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_belt | 1 | 57.018633540372676 | 55.54839081810636 | 57.018633540372676 | 1610 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_belt | 1 | 64.472049689441 | 64.00163843414423 | 64.47204968944101 | 1610 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_belt | 1 | 65.59006211180125 | 65.06727153685394 | 65.59006211180125 | 1610 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_belt | 1 | 64.59627329192547 | 63.3681141443661 | 64.59627329192547 | 1610 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_belt | 1 | 71.18012422360248 | 70.69723141447788 | 71.18012422360248 | 1610 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_belt | 1 | 71.05590062111801 | 70.60384943165286 | 71.05590062111801 | 1610 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_belt | 1 | 69.87577639751554 | 68.99828347281921 | 69.87577639751554 | 1610 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_belt | 1 | 75.3416149068323 | 75.23012180545776 | 75.3416149068323 | 1610 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_belt | 1 | 73.91304347826086 | 73.68723069490925 | 73.91304347826087 | 1610 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_belt | 1 | 76.3975155279503 | 76.07795431370218 | 76.3975155279503 | 1610 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_belt | 1 | 79.56521739130434 | 79.44043477196986 | 79.56521739130434 | 1610 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_belt | 1 | 76.77018633540372 | 76.59387036025419 | 76.77018633540372 | 1610 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_belt | 1 | 79.8136645962733 | 79.59095929019882 | 79.81366459627328 | 1610 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_belt | 1 | 82.79503105590062 | 82.68911053070511 | 82.79503105590061 | 1610 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_belt | 1 | 78.94409937888199 | 78.77839650952232 | 78.94409937888199 | 1610 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_belt | 1 | 84.72049689440993 | 84.5896965325001 | 84.72049689440995 | 1610 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_belt | 1 | 85.09316770186336 | 84.9417969769624 | 85.09316770186335 | 1610 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_belt | 1 | 79.00621118012423 | 78.84672319210904 | 79.00621118012421 | 1610 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_belt | 1 | 87.39130434782608 | 87.28735037444399 | 87.39130434782606 | 1610 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | phone_belt | 1 | 86.95652173913044 | 86.84404001154539 | 86.95652173913044 | 1610 | ok |
| harnet | 4.49 | prototype | 128 | False | False | phone_belt | 1 | 80.1863354037267 | 80.06471424136464 | 80.18633540372669 | 1610 | ok |
| harnet | 4.49 | ridge | 128 | False | False | phone_belt | 1 | 89.50310559006212 | 89.43667388149782 | 89.5031055900621 | 1610 | ok |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | watch_wrist_proxy | 1 | 55.40372670807453 | 52.61022537170182 | 55.40372670807453 | 1610 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | watch_wrist_proxy | 1 | 64.53416149068323 | 64.08175104568802 | 64.53416149068323 | 1610 | ok |
| harnet | 4.49 | prototype | 1 | False | False | watch_wrist_proxy | 1 | 64.53416149068323 | 64.08175104568802 | 64.53416149068323 | 1610 | ok |
| harnet | 4.49 | ridge | 1 | False | False | watch_wrist_proxy | 1 | 64.40993788819875 | 63.482057028995534 | 64.40993788819877 | 1610 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | watch_wrist_proxy | 1 | 69.1304347826087 | 68.90343913204623 | 69.13043478260867 | 1610 | ok |
| harnet | 4.49 | prototype | 2 | False | False | watch_wrist_proxy | 1 | 70.0 | 69.60881342434223 | 70.0 | 1610 | ok |
| harnet | 4.49 | ridge | 2 | False | False | watch_wrist_proxy | 1 | 69.00621118012423 | 68.0969349307497 | 69.00621118012423 | 1610 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | watch_wrist_proxy | 1 | 72.29813664596273 | 72.02608059801253 | 72.29813664596271 | 1610 | ok |
| harnet | 4.49 | prototype | 4 | False | False | watch_wrist_proxy | 1 | 73.29192546583852 | 73.04283567563 | 73.29192546583852 | 1610 | ok |
| harnet | 4.49 | ridge | 4 | False | False | watch_wrist_proxy | 1 | 72.17391304347827 | 71.3949015241374 | 72.17391304347825 | 1610 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | watch_wrist_proxy | 1 | 75.03105590062111 | 74.7792111892365 | 75.03105590062111 | 1610 | ok |
| harnet | 4.49 | prototype | 8 | False | False | watch_wrist_proxy | 1 | 75.527950310559 | 75.32310171837396 | 75.527950310559 | 1610 | ok |
| harnet | 4.49 | ridge | 8 | False | False | watch_wrist_proxy | 1 | 77.01863354037268 | 76.2946715194427 | 77.01863354037266 | 1610 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | watch_wrist_proxy | 1 | 76.52173913043478 | 76.37862159845386 | 76.52173913043477 | 1610 | ok |
| harnet | 4.49 | prototype | 16 | False | False | watch_wrist_proxy | 1 | 78.01242236024845 | 77.9073123666383 | 78.01242236024845 | 1610 | ok |
| harnet | 4.49 | ridge | 16 | False | False | watch_wrist_proxy | 1 | 78.88198757763976 | 78.40611041331097 | 78.88198757763976 | 1610 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | watch_wrist_proxy | 1 | 77.51552795031056 | 77.43974562507195 | 77.51552795031057 | 1610 | ok |
| harnet | 4.49 | prototype | 32 | False | False | watch_wrist_proxy | 1 | 78.57142857142857 | 78.53189325252902 | 78.57142857142856 | 1610 | ok |
| harnet | 4.49 | ridge | 32 | False | False | watch_wrist_proxy | 1 | 82.6086956521739 | 82.30700340202914 | 82.6086956521739 | 1610 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | watch_wrist_proxy | 1 | 80.62111801242236 | 80.62798550627414 | 80.62111801242236 | 1610 | ok |
| harnet | 4.49 | prototype | 64 | False | False | watch_wrist_proxy | 1 | 79.62732919254658 | 79.57029271700894 | 79.62732919254657 | 1610 | ok |
| harnet | 4.49 | ridge | 64 | False | False | watch_wrist_proxy | 1 | 83.85093167701864 | 83.67199328341209 | 83.85093167701864 | 1610 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | watch_wrist_proxy | 1 | 82.6086956521739 | 82.60206040115236 | 82.60869565217392 | 1610 | ok |
| harnet | 4.49 | prototype | 128 | False | False | watch_wrist_proxy | 1 | 80.4968944099379 | 80.45152473663423 | 80.49689440993788 | 1610 | ok |
| harnet | 4.49 | ridge | 128 | False | False | watch_wrist_proxy | 1 | 85.96273291925466 | 85.88771315812713 | 85.96273291925466 | 1610 | ok |
| harnet | 4.49 | zero_support | 0 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| harnet | 4.49 | 1nn | 1 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 69.56521739130434 | 69.25232719955649 | 69.56521739130436 | 1610 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 69.56521739130434 | 69.25232719955649 | 69.56521739130436 | 1610 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 66.52173913043478 | 65.53272181860893 | 66.52173913043478 | 1610 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 76.45962732919254 | 76.10169011107759 | 76.45962732919254 | 1610 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 75.09316770186335 | 74.62829292701943 | 75.09316770186335 | 1610 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 71.11801242236024 | 70.17783818533798 | 71.11801242236025 | 1610 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 77.70186335403727 | 77.39986714433347 | 77.70186335403727 | 1610 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 77.95031055900621 | 77.60180058665294 | 77.95031055900621 | 1610 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 75.03105590062111 | 74.43745390916335 | 75.03105590062111 | 1610 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 81.86335403726707 | 81.66347244318291 | 81.86335403726709 | 1610 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 78.07453416149069 | 77.75248056211073 | 78.07453416149069 | 1610 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 78.88198757763976 | 78.40666069372558 | 78.88198757763975 | 1610 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 83.85093167701864 | 83.60064389542146 | 83.85093167701864 | 1610 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 80.0 | 79.68316435014864 | 80.0 | 1610 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 81.67701863354037 | 81.2604191596748 | 81.67701863354037 | 1610 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 86.33540372670807 | 86.19974275035113 | 86.33540372670808 | 1610 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 81.11801242236025 | 80.93291983736003 | 81.11801242236024 | 1610 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 83.85093167701864 | 83.6274139262464 | 83.85093167701864 | 1610 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 88.32298136645963 | 88.21915438783293 | 88.32298136645962 | 1610 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 81.36645962732919 | 81.19207547810967 | 81.3664596273292 | 1610 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 85.21739130434783 | 84.91525643904782 | 85.21739130434783 | 1610 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 89.50310559006212 | 89.47543387847041 | 89.50310559006212 | 1610 | ok |
| harnet | 4.49 | prototype | 128 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 80.93167701863354 | 80.7254720867287 | 80.93167701863354 | 1610 | ok |
| harnet | 4.49 | ridge | 128 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 87.01863354037268 | 86.815692737981 | 87.01863354037266 | 1610 | ok |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_right_pocket | 1 | 45.590062111801245 | 36.101835805030156 | 45.59006211180124 | 1610 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_right_pocket | 1 | 79.44099378881988 | 79.4626367853879 | 79.44099378881988 | 1610 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_right_pocket | 1 | 79.44099378881988 | 79.4626367853879 | 79.44099378881988 | 1610 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_right_pocket | 1 | 79.06832298136646 | 79.06278677018399 | 79.06832298136646 | 1610 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_right_pocket | 1 | 87.51552795031056 | 87.52966009231845 | 87.51552795031056 | 1610 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_right_pocket | 1 | 82.73291925465838 | 82.6228591134586 | 82.73291925465838 | 1610 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_right_pocket | 1 | 83.29192546583852 | 83.18300641428822 | 83.29192546583852 | 1610 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_right_pocket | 1 | 92.29813664596274 | 92.31493331856241 | 92.29813664596273 | 1610 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_right_pocket | 1 | 85.65217391304348 | 85.27136788745148 | 85.65217391304348 | 1610 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_right_pocket | 1 | 84.96894409937889 | 84.51924700826616 | 84.96894409937887 | 1610 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_right_pocket | 1 | 94.53416149068323 | 94.53581178293705 | 94.53416149068323 | 1610 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_right_pocket | 1 | 86.08695652173914 | 85.59598483014223 | 86.08695652173914 | 1610 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_right_pocket | 1 | 86.0248447204969 | 85.43646396417873 | 86.02484472049687 | 1610 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_right_pocket | 1 | 95.46583850931677 | 95.46456373937268 | 95.46583850931675 | 1610 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_right_pocket | 1 | 88.32298136645963 | 87.77764536754572 | 88.32298136645963 | 1610 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_right_pocket | 1 | 88.69565217391305 | 88.11824897095332 | 88.69565217391305 | 1610 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_right_pocket | 1 | 95.527950310559 | 95.52589937477839 | 95.527950310559 | 1610 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_right_pocket | 1 | 87.70186335403727 | 86.92777247780458 | 87.70186335403727 | 1610 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_right_pocket | 1 | 88.26086956521739 | 87.38216113503323 | 88.26086956521742 | 1610 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_right_pocket | 1 | 97.88819875776397 | 97.89093668453732 | 97.88819875776397 | 1610 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_right_pocket | 1 | 87.45341614906832 | 86.40802679783563 | 87.45341614906833 | 1610 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_right_pocket | 1 | 88.50931677018633 | 87.51731697350459 | 88.50931677018635 | 1610 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | phone_right_pocket | 1 | 98.26086956521739 | 98.26255799387303 | 98.2608695652174 | 1610 | ok |
| unimts | 68.61 | prototype | 128 | True | False | phone_right_pocket | 1 | 87.63975155279503 | 86.6716207564463 | 87.63975155279503 | 1610 | ok |
| unimts | 68.61 | ridge | 128 | True | False | phone_right_pocket | 1 | 90.0 | 89.39767151260975 | 90.00000000000001 | 1610 | ok |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_left_pocket | 1 | 45.27950310559007 | 35.74893971169741 | 45.27950310559007 | 1610 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_left_pocket | 1 | 80.4968944099379 | 80.4849266760234 | 80.4968944099379 | 1610 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_left_pocket | 1 | 80.4968944099379 | 80.4849266760234 | 80.4968944099379 | 1610 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_left_pocket | 1 | 79.19254658385093 | 79.0506795041486 | 79.19254658385093 | 1610 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_left_pocket | 1 | 88.07453416149067 | 88.09407822789544 | 88.0745341614907 | 1610 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_left_pocket | 1 | 86.0248447204969 | 85.98172440251142 | 86.0248447204969 | 1610 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_left_pocket | 1 | 85.46583850931677 | 85.27384701279631 | 85.46583850931677 | 1610 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_left_pocket | 1 | 92.11180124223603 | 92.10143599740573 | 92.11180124223601 | 1610 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_left_pocket | 1 | 86.52173913043478 | 86.24829021207871 | 86.52173913043477 | 1610 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_left_pocket | 1 | 86.7080745341615 | 86.36303311064408 | 86.7080745341615 | 1610 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_left_pocket | 1 | 94.472049689441 | 94.46221258170605 | 94.472049689441 | 1610 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_left_pocket | 1 | 86.45962732919254 | 86.05895209866695 | 86.45962732919253 | 1610 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_left_pocket | 1 | 85.527950310559 | 85.04557559742733 | 85.52795031055899 | 1610 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_left_pocket | 1 | 95.34161490683229 | 95.34558443665858 | 95.34161490683229 | 1610 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_left_pocket | 1 | 88.81987577639751 | 88.39514721115464 | 88.81987577639751 | 1610 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_left_pocket | 1 | 89.00621118012423 | 88.49364172533375 | 89.00621118012421 | 1610 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_left_pocket | 1 | 95.77639751552796 | 95.77674676967341 | 95.77639751552796 | 1610 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_left_pocket | 1 | 88.38509316770187 | 87.70726460672552 | 88.38509316770187 | 1610 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_left_pocket | 1 | 89.56521739130436 | 88.98566335213226 | 89.56521739130434 | 1610 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_left_pocket | 1 | 96.64596273291926 | 96.63520947214866 | 96.64596273291926 | 1610 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_left_pocket | 1 | 88.4472049689441 | 87.68776954573575 | 88.44720496894409 | 1610 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_left_pocket | 1 | 90.31055900621118 | 89.82261774989476 | 90.3105590062112 | 1610 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | phone_left_pocket | 1 | 96.8944099378882 | 96.88544726740213 | 96.89440993788821 | 1610 | ok |
| unimts | 68.61 | prototype | 128 | True | False | phone_left_pocket | 1 | 88.19875776397515 | 87.45057669555634 | 88.19875776397515 | 1610 | ok |
| unimts | 68.61 | ridge | 128 | True | False | phone_left_pocket | 1 | 91.36645962732919 | 91.09383902116512 | 91.36645962732919 | 1610 | ok |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_belt | 1 | 50.745341614906835 | 41.43632211245663 | 50.745341614906835 | 1610 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_belt | 1 | 69.68944099378882 | 69.55594617504414 | 69.68944099378882 | 1610 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_belt | 1 | 69.68944099378882 | 69.55594617504414 | 69.68944099378882 | 1610 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_belt | 1 | 67.5776397515528 | 67.13897131388335 | 67.5776397515528 | 1610 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_belt | 1 | 77.76397515527951 | 77.66649972723813 | 77.7639751552795 | 1610 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_belt | 1 | 76.77018633540372 | 76.6921755362478 | 76.77018633540372 | 1610 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_belt | 1 | 74.09937888198758 | 73.79504421642118 | 74.09937888198758 | 1610 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_belt | 1 | 84.22360248447205 | 84.15901128879787 | 84.22360248447205 | 1610 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_belt | 1 | 79.13043478260869 | 79.1130434969928 | 79.13043478260869 | 1610 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_belt | 1 | 77.51552795031056 | 77.23474591760873 | 77.51552795031057 | 1610 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_belt | 1 | 84.53416149068323 | 84.45965282553229 | 84.53416149068323 | 1610 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_belt | 1 | 79.75155279503106 | 79.73070721441273 | 79.75155279503106 | 1610 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_belt | 1 | 79.75155279503106 | 79.50457725770444 | 79.75155279503106 | 1610 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_belt | 1 | 87.39130434782608 | 87.31584858852138 | 87.3913043478261 | 1610 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_belt | 1 | 82.79503105590062 | 82.76828300768769 | 82.79503105590062 | 1610 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_belt | 1 | 85.15527950310559 | 85.00687553518178 | 85.15527950310559 | 1610 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_belt | 1 | 89.37888198757764 | 89.33777376249948 | 89.37888198757766 | 1610 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_belt | 1 | 84.03726708074534 | 84.01789718779973 | 84.03726708074534 | 1610 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_belt | 1 | 86.83229813664596 | 86.6740010763076 | 86.83229813664596 | 1610 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_belt | 1 | 90.55900621118013 | 90.49894648157452 | 90.55900621118012 | 1610 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_belt | 1 | 85.77639751552795 | 85.75090521369223 | 85.77639751552793 | 1610 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_belt | 1 | 90.12422360248448 | 90.04204977760678 | 90.12422360248448 | 1610 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | phone_belt | 1 | 90.8695652173913 | 90.83544366407683 | 90.86956521739128 | 1610 | ok |
| unimts | 68.61 | prototype | 128 | True | False | phone_belt | 1 | 85.09316770186336 | 85.06159349034513 | 85.09316770186335 | 1610 | ok |
| unimts | 68.61 | ridge | 128 | True | False | phone_belt | 1 | 91.11801242236025 | 91.08598935726654 | 91.11801242236025 | 1610 | ok |
| unimts | 68.61 | native_zero_support | 0 | True | False | watch_wrist_proxy | 1 | 45.40372670807453 | 36.95132027877782 | 45.40372670807454 | 1610 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | watch_wrist_proxy | 1 | 65.65217391304347 | 65.88369119193479 | 65.65217391304347 | 1610 | ok |
| unimts | 68.61 | prototype | 1 | True | False | watch_wrist_proxy | 1 | 65.65217391304347 | 65.88369119193479 | 65.65217391304347 | 1610 | ok |
| unimts | 68.61 | ridge | 1 | True | False | watch_wrist_proxy | 1 | 62.2360248447205 | 61.51743134003327 | 62.2360248447205 | 1610 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | watch_wrist_proxy | 1 | 73.91304347826086 | 74.03849704519843 | 73.91304347826087 | 1610 | ok |
| unimts | 68.61 | prototype | 2 | True | False | watch_wrist_proxy | 1 | 69.19254658385093 | 69.46077123381177 | 69.19254658385093 | 1610 | ok |
| unimts | 68.61 | ridge | 2 | True | False | watch_wrist_proxy | 1 | 66.14906832298136 | 65.84276699762361 | 66.14906832298138 | 1610 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | watch_wrist_proxy | 1 | 78.63354037267081 | 78.82411596312656 | 78.63354037267081 | 1610 | ok |
| unimts | 68.61 | prototype | 4 | True | False | watch_wrist_proxy | 1 | 72.73291925465838 | 72.90228084076578 | 72.73291925465838 | 1610 | ok |
| unimts | 68.61 | ridge | 4 | True | False | watch_wrist_proxy | 1 | 70.80745341614907 | 70.7126262326236 | 70.80745341614907 | 1610 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | watch_wrist_proxy | 1 | 81.92546583850931 | 82.05793170399284 | 81.92546583850931 | 1610 | ok |
| unimts | 68.61 | prototype | 8 | True | False | watch_wrist_proxy | 1 | 71.92546583850931 | 71.99124628954249 | 71.92546583850931 | 1610 | ok |
| unimts | 68.61 | ridge | 8 | True | False | watch_wrist_proxy | 1 | 72.67080745341616 | 72.44382804005491 | 72.67080745341615 | 1610 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | watch_wrist_proxy | 1 | 84.96894409937889 | 85.04196881140054 | 84.96894409937889 | 1610 | ok |
| unimts | 68.61 | prototype | 16 | True | False | watch_wrist_proxy | 1 | 75.3416149068323 | 75.4092725129884 | 75.34161490683229 | 1610 | ok |
| unimts | 68.61 | ridge | 16 | True | False | watch_wrist_proxy | 1 | 75.90062111801242 | 75.69384828057586 | 75.90062111801241 | 1610 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | watch_wrist_proxy | 1 | 86.52173913043478 | 86.61347647794376 | 86.52173913043478 | 1610 | ok |
| unimts | 68.61 | prototype | 32 | True | False | watch_wrist_proxy | 1 | 75.27950310559007 | 75.32721860186913 | 75.27950310559007 | 1610 | ok |
| unimts | 68.61 | ridge | 32 | True | False | watch_wrist_proxy | 1 | 78.44720496894409 | 78.24568784661119 | 78.4472049689441 | 1610 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | watch_wrist_proxy | 1 | 88.38509316770187 | 88.46575738905597 | 88.38509316770187 | 1610 | ok |
| unimts | 68.61 | prototype | 64 | True | False | watch_wrist_proxy | 1 | 76.52173913043478 | 76.55216355218253 | 76.52173913043478 | 1610 | ok |
| unimts | 68.61 | ridge | 64 | True | False | watch_wrist_proxy | 1 | 81.73913043478261 | 81.64628372451948 | 81.73913043478261 | 1610 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | watch_wrist_proxy | 1 | 88.01242236024845 | 88.08760949635034 | 88.01242236024845 | 1610 | ok |
| unimts | 68.61 | prototype | 128 | True | False | watch_wrist_proxy | 1 | 77.20496894409938 | 77.15306407754505 | 77.20496894409938 | 1610 | ok |
| unimts | 68.61 | ridge | 128 | True | False | watch_wrist_proxy | 1 | 86.27329192546583 | 86.22015306241414 | 86.27329192546583 | 1610 | ok |
| unimts | 68.61 | zero_support | 0 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| unimts | 68.61 | 1nn | 1 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 77.88819875776397 | 77.9940691295079 | 77.88819875776397 | 1610 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 77.88819875776397 | 77.9940691295079 | 77.88819875776397 | 1610 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 78.26086956521739 | 78.15898817286114 | 78.26086956521739 | 1610 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 85.71428571428571 | 85.72956667780501 | 85.71428571428571 | 1610 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 82.85714285714286 | 82.89778611848637 | 82.85714285714286 | 1610 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 82.67080745341615 | 82.59398335060085 | 82.67080745341615 | 1610 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 90.8695652173913 | 90.89144990578755 | 90.86956521739131 | 1610 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 84.03726708074534 | 84.02356440287377 | 84.03726708074534 | 1610 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 84.59627329192546 | 84.5703802293098 | 84.59627329192546 | 1610 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 92.2360248447205 | 92.2560594555993 | 92.2360248447205 | 1610 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 84.16149068322981 | 83.9593386055205 | 84.16149068322981 | 1610 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 86.27329192546583 | 86.06524600773454 | 86.27329192546584 | 1610 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 94.09937888198758 | 94.11194825813409 | 94.09937888198758 | 1610 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 84.84472049689441 | 84.62525756039963 | 84.84472049689442 | 1610 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 88.38509316770187 | 88.23467402845903 | 88.38509316770187 | 1610 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 95.03105590062113 | 95.03808839876793 | 95.03105590062111 | 1610 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 85.527950310559 | 85.16069358491085 | 85.527950310559 | 1610 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 90.4968944099379 | 90.3673454485422 | 90.4968944099379 | 1610 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 95.90062111801242 | 95.90625676146442 | 95.90062111801242 | 1610 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 85.21739130434783 | 84.73228211024869 | 85.21739130434783 | 1610 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 91.24223602484473 | 91.14363140388824 | 91.24223602484473 | 1610 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 96.52173913043478 | 96.52792214368918 | 96.5217391304348 | 1610 | ok |
| unimts | 68.61 | prototype | 128 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 84.22360248447205 | 83.7201656620153 | 84.22360248447205 | 1610 | ok |
| unimts | 68.61 | ridge | 128 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 94.72049689440993 | 94.72397179843229 | 94.72049689440996 | 1610 | ok |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_right_pocket | 1 | 14.285714285714285 | 3.571428571428571 | 14.285714285714285 | 1610 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_right_pocket | 1 | 21.428571428571427 | 21.5393951656335 | 21.428571428571427 | 1610 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_right_pocket | 1 | 21.428571428571427 | 21.5393951656335 | 21.428571428571427 | 1610 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_right_pocket | 1 | 22.981366459627328 | 22.874881240731845 | 22.98136645962733 | 1610 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_right_pocket | 1 | 26.08695652173913 | 26.455925449618356 | 26.086956521739125 | 1610 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_right_pocket | 1 | 23.975155279503106 | 24.131638535757148 | 23.975155279503106 | 1610 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_right_pocket | 1 | 23.664596273291927 | 23.595312334043506 | 23.664596273291924 | 1610 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_right_pocket | 1 | 29.565217391304348 | 29.958308862499994 | 29.565217391304344 | 1610 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_right_pocket | 1 | 26.77018633540373 | 26.78049023531249 | 26.77018633540373 | 1610 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_right_pocket | 1 | 23.91304347826087 | 23.27106836685052 | 23.913043478260867 | 1610 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_right_pocket | 1 | 34.84472049689441 | 35.37871290641231 | 34.84472049689441 | 1610 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_right_pocket | 1 | 27.82608695652174 | 27.60387184102962 | 27.82608695652174 | 1610 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_right_pocket | 1 | 26.83229813664596 | 25.709065217006028 | 26.832298136645967 | 1610 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_right_pocket | 1 | 40.37267080745342 | 41.02663632736583 | 40.37267080745342 | 1610 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_right_pocket | 1 | 30.31055900621118 | 30.139191734100006 | 30.310559006211186 | 1610 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_right_pocket | 1 | 28.633540372670808 | 27.08844860224293 | 28.6335403726708 | 1610 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_right_pocket | 1 | 45.65217391304348 | 46.4132271914769 | 45.65217391304348 | 1610 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_right_pocket | 1 | 29.503105590062113 | 29.043064398877043 | 29.503105590062113 | 1610 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_right_pocket | 1 | 33.1055900621118 | 30.859580531272297 | 33.105590062111794 | 1610 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_right_pocket | 1 | 50.18633540372671 | 50.72346921807716 | 50.18633540372671 | 1610 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_right_pocket | 1 | 30.99378881987578 | 30.297809534036258 | 30.99378881987578 | 1610 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_right_pocket | 1 | 38.69565217391304 | 35.75821043736207 | 38.69565217391305 | 1610 | ok |
| normwear | 1293.86 | 1nn | 128 | True | False | phone_right_pocket | 1 | 53.85093167701863 | 54.482513485677295 | 53.85093167701863 | 1610 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | phone_right_pocket | 1 | 30.186335403726712 | 28.788075789810343 | 30.186335403726712 | 1610 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | phone_right_pocket | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_left_pocket | 1 | 14.285714285714285 | 3.571428571428571 | 14.285714285714285 | 1610 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_left_pocket | 1 | 25.71428571428571 | 25.97863833613 | 25.71428571428571 | 1610 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_left_pocket | 1 | 25.71428571428571 | 25.97863833613 | 25.71428571428571 | 1610 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_left_pocket | 1 | 24.03726708074534 | 23.235716213484594 | 24.03726708074534 | 1610 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_left_pocket | 1 | 27.32919254658385 | 27.707716994365462 | 27.32919254658385 | 1610 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_left_pocket | 1 | 26.583850931677016 | 26.734757336261488 | 26.583850931677024 | 1610 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_left_pocket | 1 | 24.658385093167702 | 23.676286444309614 | 24.658385093167706 | 1610 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_left_pocket | 1 | 31.11801242236025 | 31.795819460123248 | 31.11801242236025 | 1610 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_left_pocket | 1 | 27.82608695652174 | 27.62444387157496 | 27.82608695652174 | 1610 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_left_pocket | 1 | 24.53416149068323 | 22.47982298403336 | 24.534161490683232 | 1610 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_left_pocket | 1 | 38.32298136645963 | 39.11710641621723 | 38.32298136645963 | 1610 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_left_pocket | 1 | 31.67701863354037 | 31.05086057285415 | 31.677018633540367 | 1610 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_left_pocket | 1 | 27.82608695652174 | 24.75849289206869 | 27.82608695652174 | 1610 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_left_pocket | 1 | 43.1055900621118 | 44.01382606697739 | 43.105590062111816 | 1610 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_left_pocket | 1 | 31.801242236024844 | 30.52294611086505 | 31.801242236024844 | 1610 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_left_pocket | 1 | 30.99378881987578 | 26.201285579197126 | 30.99378881987578 | 1610 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_left_pocket | 1 | 47.329192546583855 | 47.95557192241343 | 47.32919254658385 | 1610 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_left_pocket | 1 | 33.04347826086956 | 31.33248660600921 | 33.04347826086956 | 1610 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_left_pocket | 1 | 32.17391304347826 | 26.185992677969182 | 32.17391304347825 | 1610 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_left_pocket | 1 | 52.17391304347826 | 52.698303166715355 | 52.17391304347826 | 1610 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_left_pocket | 1 | 34.223602484472046 | 32.2187979075587 | 34.223602484472046 | 1610 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_left_pocket | 1 | 37.70186335403727 | 32.02086233636036 | 37.701863354037265 | 1610 | ok |
| normwear | 1293.86 | 1nn | 128 | True | False | phone_left_pocket | 1 | 54.8447204968944 | 55.61171304506524 | 54.8447204968944 | 1610 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | phone_left_pocket | 1 | 33.66459627329193 | 31.01139765552151 | 33.66459627329192 | 1610 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | phone_left_pocket | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_belt | 1 | 14.285714285714285 | 3.571428571428571 | 14.285714285714285 | 1610 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_belt | 1 | 23.726708074534162 | 23.742609250331668 | 23.726708074534166 | 1610 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_belt | 1 | 23.726708074534162 | 23.742609250331668 | 23.726708074534166 | 1610 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_belt | 1 | 23.354037267080745 | 23.01620847488353 | 23.354037267080745 | 1610 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_belt | 1 | 26.39751552795031 | 26.447888555225983 | 26.397515527950315 | 1610 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_belt | 1 | 26.583850931677016 | 26.58189640016735 | 26.583850931677024 | 1610 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_belt | 1 | 26.335403726708073 | 25.35389817923166 | 26.335403726708073 | 1610 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_belt | 1 | 30.248447204968944 | 30.27257924844884 | 30.248447204968937 | 1610 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_belt | 1 | 25.279503105590063 | 25.0471069249951 | 25.279503105590063 | 1610 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_belt | 1 | 24.53416149068323 | 22.77117399924343 | 24.534161490683232 | 1610 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_belt | 1 | 35.09316770186335 | 35.212179284705776 | 35.09316770186335 | 1610 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_belt | 1 | 27.639751552795033 | 27.278590741039178 | 27.639751552795033 | 1610 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_belt | 1 | 25.71428571428571 | 23.43218577315529 | 25.71428571428572 | 1610 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_belt | 1 | 38.38509316770186 | 38.51665763829648 | 38.38509316770186 | 1610 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_belt | 1 | 28.819875776397513 | 28.530368827251156 | 28.819875776397513 | 1610 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_belt | 1 | 27.888198757763977 | 25.605021326184 | 27.888198757763977 | 1610 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_belt | 1 | 43.41614906832298 | 43.49764438987982 | 43.41614906832298 | 1610 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_belt | 1 | 29.689440993788818 | 29.563721185085555 | 29.689440993788818 | 1610 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_belt | 1 | 29.130434782608695 | 26.514404493828742 | 29.1304347826087 | 1610 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_belt | 1 | 45.65217391304348 | 45.60800377559154 | 45.652173913043484 | 1610 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_belt | 1 | 29.503105590062113 | 28.802083730614473 | 29.503105590062113 | 1610 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_belt | 1 | 33.54037267080746 | 31.19122318309877 | 33.54037267080745 | 1610 | ok |
| normwear | 1293.86 | 1nn | 128 | True | False | phone_belt | 1 | 48.75776397515528 | 48.76095974529623 | 48.757763975155285 | 1610 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | phone_belt | 1 | 29.440993788819874 | 28.939436150556798 | 29.440993788819874 | 1610 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | phone_belt | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | watch_wrist_proxy | 1 | 14.285714285714285 | 3.571428571428571 | 14.285714285714285 | 1610 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | watch_wrist_proxy | 1 | 30.372670807453417 | 30.01143625452564 | 30.372670807453417 | 1610 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | watch_wrist_proxy | 1 | 30.372670807453417 | 30.01143625452564 | 30.372670807453417 | 1610 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | watch_wrist_proxy | 1 | 27.267080745341616 | 25.570098915841815 | 27.267080745341616 | 1610 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | watch_wrist_proxy | 1 | 34.47204968944099 | 34.191990137451405 | 34.47204968944099 | 1610 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | watch_wrist_proxy | 1 | 33.29192546583851 | 32.819446140812055 | 33.29192546583851 | 1610 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | watch_wrist_proxy | 1 | 29.19254658385093 | 26.075249725411282 | 29.192546583850927 | 1610 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | watch_wrist_proxy | 1 | 41.49068322981366 | 41.346803310462796 | 41.490683229813676 | 1610 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | watch_wrist_proxy | 1 | 37.95031055900621 | 36.708837183264095 | 37.95031055900621 | 1610 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | watch_wrist_proxy | 1 | 34.96894409937889 | 29.429823918804537 | 34.96894409937888 | 1610 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | watch_wrist_proxy | 1 | 43.975155279503106 | 43.83812232761042 | 43.975155279503106 | 1610 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | watch_wrist_proxy | 1 | 40.62111801242236 | 38.549055964912824 | 40.62111801242236 | 1610 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | watch_wrist_proxy | 1 | 37.204968944099384 | 29.260593642121275 | 37.204968944099384 | 1610 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | watch_wrist_proxy | 1 | 47.329192546583855 | 47.32192282868349 | 47.32919254658384 | 1610 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | watch_wrist_proxy | 1 | 42.670807453416145 | 40.553427417848994 | 42.67080745341615 | 1610 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | watch_wrist_proxy | 1 | 37.515527950310556 | 27.248361366760577 | 37.515527950310556 | 1610 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | watch_wrist_proxy | 1 | 53.54037267080746 | 53.34209132242961 | 53.54037267080744 | 1610 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | watch_wrist_proxy | 1 | 43.60248447204969 | 41.097089929517516 | 43.60248447204969 | 1610 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | watch_wrist_proxy | 1 | 39.254658385093165 | 28.306429420659008 | 39.25465838509317 | 1610 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | watch_wrist_proxy | 1 | 56.024844720496894 | 55.970017044525676 | 56.024844720496894 | 1610 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | watch_wrist_proxy | 1 | 43.04347826086957 | 40.325718857915724 | 43.04347826086957 | 1610 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | watch_wrist_proxy | 1 | 41.36645962732919 | 31.319690956576952 | 41.36645962732919 | 1610 | ok |
| normwear | 1293.86 | 1nn | 128 | True | False | watch_wrist_proxy | 1 | 58.69565217391305 | 58.54176631379859 | 58.69565217391305 | 1610 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | watch_wrist_proxy | 1 | 43.975155279503106 | 41.42960088982606 | 43.975155279503106 | 1610 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | watch_wrist_proxy | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | zero_support | 0 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 27.763975155279503 | 27.853931436753104 | 27.763975155279503 | 1610 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 27.763975155279503 | 27.853931436753104 | 27.763975155279503 | 1610 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 26.8944099378882 | 26.409118309934165 | 26.8944099378882 | 1610 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 35.900621118012424 | 36.05535530247925 | 35.90062111801242 | 1610 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 32.17391304347826 | 32.43714405190857 | 32.17391304347826 | 1610 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 27.888198757763977 | 26.967780894980383 | 27.88819875776397 | 1610 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 39.19254658385093 | 39.40535164461303 | 39.19254658385093 | 1610 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 34.161490683229815 | 34.267530353108945 | 34.161490683229815 | 1610 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 28.074534161490682 | 27.012134382825465 | 28.074534161490682 | 1610 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 43.7888198757764 | 44.067681361736646 | 43.7888198757764 | 1610 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 35.3416149068323 | 35.27506814325915 | 35.341614906832305 | 1610 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 33.47826086956522 | 31.434950641579277 | 33.47826086956522 | 1610 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 50.931677018633536 | 51.09017078806046 | 50.931677018633536 | 1610 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 37.70186335403727 | 37.65623877635389 | 37.701863354037265 | 1610 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 34.84472049689441 | 31.591059981488296 | 34.84472049689442 | 1610 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 58.26086956521739 | 58.34573209331053 | 58.26086956521739 | 1610 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 39.75155279503105 | 39.54518481161763 | 39.75155279503105 | 1610 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 37.453416149068325 | 34.09004643194316 | 37.45341614906832 | 1610 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 62.2360248447205 | 62.27337598239241 | 62.2360248447205 | 1610 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 41.24223602484472 | 40.90498759366996 | 41.24223602484472 | 1610 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 40.18633540372671 | 37.08675876493132 | 40.186335403726716 | 1610 | ok |
| normwear | 1293.86 | 1nn | 128 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 66.14906832298136 | 66.08517813635885 | 66.14906832298136 | 1610 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 42.04968944099379 | 41.86051428802828 | 42.04968944099378 | 1610 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_right_pocket | 1 | 41.42857142857143 | 34.614026975618316 | 41.42857142857142 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_right_pocket | 1 | 71.55279503105591 | 71.79856868084549 | 71.5527950310559 | 1610 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_right_pocket | 1 | 71.55279503105591 | 71.79856868084549 | 71.5527950310559 | 1610 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_right_pocket | 1 | 65.46583850931677 | 62.84149864710551 | 65.46583850931677 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_right_pocket | 1 | 78.13664596273291 | 78.30594054376293 | 78.13664596273291 | 1610 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_right_pocket | 1 | 77.14285714285715 | 77.28279394311075 | 77.14285714285714 | 1610 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_right_pocket | 1 | 67.08074534161491 | 64.4803298478128 | 67.08074534161491 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_right_pocket | 1 | 86.3975155279503 | 86.597259876703 | 86.3975155279503 | 1610 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_right_pocket | 1 | 82.79503105590062 | 82.94876125736354 | 82.79503105590062 | 1610 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_right_pocket | 1 | 70.4968944099379 | 68.05755169472253 | 70.4968944099379 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_right_pocket | 1 | 91.49068322981367 | 91.51768559050018 | 91.49068322981365 | 1610 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_right_pocket | 1 | 88.4472049689441 | 88.56310481968652 | 88.44720496894409 | 1610 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_right_pocket | 1 | 75.77639751552795 | 73.86053176083654 | 75.77639751552796 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_right_pocket | 1 | 91.73913043478261 | 91.73632488173055 | 91.73913043478261 | 1610 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_right_pocket | 1 | 91.98757763975155 | 92.03715888545658 | 91.98757763975155 | 1610 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_right_pocket | 1 | 80.74534161490683 | 79.28753262646818 | 80.74534161490683 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_right_pocket | 1 | 92.2360248447205 | 92.24289533471651 | 92.2360248447205 | 1610 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_right_pocket | 1 | 91.98757763975155 | 92.01438061966864 | 91.98757763975156 | 1610 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_right_pocket | 1 | 84.72049689440993 | 83.89205688844696 | 84.72049689440995 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_right_pocket | 1 | 91.80124223602485 | 91.80472030525733 | 91.80124223602485 | 1610 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_right_pocket | 1 | 93.47826086956522 | 93.50293970602733 | 93.4782608695652 | 1610 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_right_pocket | 1 | 87.01863354037268 | 86.47318559991552 | 87.01863354037268 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 128 | False | False | phone_right_pocket | 1 | 91.73913043478261 | 91.72329239149008 | 91.73913043478261 | 1610 | ok |
| limubert_x | 0.055 | prototype | 128 | False | False | phone_right_pocket | 1 | 93.47826086956522 | 93.4805924832643 | 93.47826086956522 | 1610 | ok |
| limubert_x | 0.055 | ridge | 128 | False | False | phone_right_pocket | 1 | 89.81366459627328 | 89.5031103241317 | 89.81366459627328 | 1610 | ok |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_left_pocket | 1 | 36.83229813664596 | 28.75906219669523 | 36.83229813664596 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_left_pocket | 1 | 76.58385093167702 | 76.98534997463075 | 76.58385093167702 | 1610 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_left_pocket | 1 | 76.58385093167702 | 76.98534997463075 | 76.58385093167702 | 1610 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_left_pocket | 1 | 65.27950310559007 | 62.64543550268539 | 65.27950310559007 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_left_pocket | 1 | 85.59006211180125 | 85.7408249741253 | 85.59006211180123 | 1610 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_left_pocket | 1 | 82.54658385093168 | 82.81722660472593 | 82.54658385093167 | 1610 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_left_pocket | 1 | 67.51552795031056 | 64.48115803883513 | 67.51552795031056 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_left_pocket | 1 | 88.63354037267081 | 88.70907632517483 | 88.63354037267081 | 1610 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_left_pocket | 1 | 86.8944099378882 | 87.02758667156758 | 86.8944099378882 | 1610 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_left_pocket | 1 | 69.93788819875778 | 66.78102764795409 | 69.93788819875776 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_left_pocket | 1 | 88.26086956521739 | 88.33438389795683 | 88.2608695652174 | 1610 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_left_pocket | 1 | 88.32298136645963 | 88.37454481999694 | 88.32298136645963 | 1610 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_left_pocket | 1 | 75.15527950310559 | 73.16011888085258 | 75.1552795031056 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_left_pocket | 1 | 90.74534161490682 | 90.84604521882288 | 90.74534161490683 | 1610 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_left_pocket | 1 | 89.75155279503106 | 89.77892458469182 | 89.75155279503106 | 1610 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_left_pocket | 1 | 81.55279503105591 | 80.48613189801088 | 81.55279503105591 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_left_pocket | 1 | 90.62111801242236 | 90.67570505219845 | 90.62111801242236 | 1610 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_left_pocket | 1 | 91.24223602484473 | 91.25876262073955 | 91.24223602484471 | 1610 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_left_pocket | 1 | 86.8944099378882 | 86.2847493451991 | 86.89440993788821 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_left_pocket | 1 | 90.74534161490682 | 90.83475315821883 | 90.74534161490682 | 1610 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_left_pocket | 1 | 90.74534161490682 | 90.75758795348223 | 90.74534161490682 | 1610 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_left_pocket | 1 | 88.13664596273291 | 87.71420902380108 | 88.13664596273291 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 128 | False | False | phone_left_pocket | 1 | 91.86335403726707 | 91.89331859194246 | 91.86335403726709 | 1610 | ok |
| limubert_x | 0.055 | prototype | 128 | False | False | phone_left_pocket | 1 | 91.49068322981367 | 91.5023881196068 | 91.49068322981367 | 1610 | ok |
| limubert_x | 0.055 | ridge | 128 | False | False | phone_left_pocket | 1 | 89.19254658385093 | 88.88739127486876 | 89.19254658385093 | 1610 | ok |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_belt | 1 | 37.88819875776397 | 32.92534220017577 | 37.88819875776398 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_belt | 1 | 62.795031055900616 | 63.15835633743101 | 62.795031055900616 | 1610 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_belt | 1 | 62.795031055900616 | 63.15835633743101 | 62.795031055900616 | 1610 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_belt | 1 | 55.21739130434783 | 52.20177021827418 | 55.21739130434783 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_belt | 1 | 72.60869565217392 | 72.75878985481768 | 72.60869565217392 | 1610 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_belt | 1 | 61.24223602484472 | 61.750196313587544 | 61.24223602484473 | 1610 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_belt | 1 | 54.90683229813664 | 51.86376549643209 | 54.90683229813664 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_belt | 1 | 74.84472049689441 | 75.11317359280743 | 74.84472049689442 | 1610 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_belt | 1 | 58.07453416149069 | 59.002376062565865 | 58.074534161490675 | 1610 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_belt | 1 | 55.40372670807453 | 51.72063242840723 | 55.40372670807453 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_belt | 1 | 77.51552795031056 | 77.78403192176715 | 77.51552795031056 | 1610 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_belt | 1 | 64.472049689441 | 64.98842900916445 | 64.472049689441 | 1610 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_belt | 1 | 61.98757763975156 | 55.97253572317696 | 61.98757763975156 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_belt | 1 | 78.13664596273291 | 78.59658811727951 | 78.13664596273291 | 1610 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_belt | 1 | 67.20496894409938 | 67.64702721180046 | 67.20496894409938 | 1610 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_belt | 1 | 64.78260869565217 | 57.46605439985012 | 64.78260869565217 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_belt | 1 | 77.76397515527951 | 78.1108418125487 | 77.7639751552795 | 1610 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_belt | 1 | 71.73913043478261 | 71.85931974989398 | 71.73913043478261 | 1610 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_belt | 1 | 65.40372670807454 | 56.99361230375521 | 65.40372670807454 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_belt | 1 | 77.26708074534162 | 77.61576353166103 | 77.26708074534162 | 1610 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_belt | 1 | 75.90062111801242 | 75.80549084412908 | 75.90062111801242 | 1610 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_belt | 1 | 65.71428571428571 | 57.39053142615206 | 65.71428571428571 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 128 | False | False | phone_belt | 1 | 77.14285714285715 | 77.460641337614 | 77.14285714285715 | 1610 | ok |
| limubert_x | 0.055 | prototype | 128 | False | False | phone_belt | 1 | 77.95031055900621 | 77.81746882889712 | 77.95031055900623 | 1610 | ok |
| limubert_x | 0.055 | ridge | 128 | False | False | phone_belt | 1 | 66.83229813664596 | 59.37320505544914 | 66.83229813664596 | 1610 | ok |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | watch_wrist_proxy | 1 | 23.229813664596275 | 19.485229353364755 | 23.22981366459627 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | watch_wrist_proxy | 1 | 67.08074534161491 | 66.72721844462319 | 67.0807453416149 | 1610 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | watch_wrist_proxy | 1 | 67.08074534161491 | 66.72721844462319 | 67.0807453416149 | 1610 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | watch_wrist_proxy | 1 | 61.67701863354037 | 59.419738510629635 | 61.67701863354037 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | watch_wrist_proxy | 1 | 75.3416149068323 | 75.34667587524717 | 75.3416149068323 | 1610 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | watch_wrist_proxy | 1 | 72.60869565217392 | 72.34999876175863 | 72.6086956521739 | 1610 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | watch_wrist_proxy | 1 | 65.83850931677019 | 64.08805930115533 | 65.83850931677019 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | watch_wrist_proxy | 1 | 80.31055900621118 | 80.34681570031049 | 80.31055900621118 | 1610 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | watch_wrist_proxy | 1 | 75.15527950310559 | 74.85962023422628 | 75.15527950310558 | 1610 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | watch_wrist_proxy | 1 | 70.62111801242236 | 69.19898935917888 | 70.62111801242236 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | watch_wrist_proxy | 1 | 84.09937888198758 | 84.20224296927097 | 84.09937888198756 | 1610 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | watch_wrist_proxy | 1 | 77.88819875776397 | 77.80243897602942 | 77.88819875776396 | 1610 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | watch_wrist_proxy | 1 | 76.58385093167702 | 75.65387675820344 | 76.58385093167703 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | watch_wrist_proxy | 1 | 87.51552795031056 | 87.5416187506373 | 87.51552795031056 | 1610 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | watch_wrist_proxy | 1 | 79.56521739130434 | 79.46357350892221 | 79.56521739130436 | 1610 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | watch_wrist_proxy | 1 | 80.93167701863354 | 80.31478835524946 | 80.93167701863354 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | watch_wrist_proxy | 1 | 89.00621118012423 | 89.05836061721051 | 89.00621118012423 | 1610 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | watch_wrist_proxy | 1 | 80.62111801242236 | 80.49746493590746 | 80.62111801242236 | 1610 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | watch_wrist_proxy | 1 | 84.59627329192546 | 84.369034655773 | 84.59627329192546 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | watch_wrist_proxy | 1 | 88.94409937888199 | 88.9885922848828 | 88.94409937888199 | 1610 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | watch_wrist_proxy | 1 | 81.98757763975155 | 81.87458100926638 | 81.98757763975155 | 1610 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | watch_wrist_proxy | 1 | 86.8944099378882 | 86.67529394611674 | 86.89440993788821 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 128 | False | False | watch_wrist_proxy | 1 | 87.5776397515528 | 87.59688905491518 | 87.5776397515528 | 1610 | ok |
| limubert_x | 0.055 | prototype | 128 | False | False | watch_wrist_proxy | 1 | 80.93167701863354 | 80.7812960732565 | 80.93167701863354 | 1610 | ok |
| limubert_x | 0.055 | ridge | 128 | False | False | watch_wrist_proxy | 1 | 88.88198757763975 | 88.71025713153723 | 88.88198757763975 | 1610 | ok |
| limubert_x | 0.055 | zero_support | 0 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 82.2360248447205 | 82.62063218128998 | 82.23602484472049 | 1610 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 82.2360248447205 | 82.62063218128998 | 82.23602484472049 | 1610 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 66.64596273291924 | 62.28193157236467 | 66.64596273291924 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 88.26086956521739 | 88.46604210447093 | 88.26086956521738 | 1610 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 86.64596273291926 | 86.9167343063886 | 86.64596273291926 | 1610 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 67.70186335403726 | 62.4373879807514 | 67.70186335403726 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 90.06211180124224 | 90.14116432374003 | 90.06211180124222 | 1610 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 90.31055900621118 | 90.41320137899405 | 90.3105590062112 | 1610 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 68.07453416149069 | 62.181397243695955 | 68.07453416149069 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 91.36645962732919 | 91.39620159382277 | 91.3664596273292 | 1610 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 92.2360248447205 | 92.31141671806172 | 92.2360248447205 | 1610 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 70.99378881987577 | 64.59498710336578 | 70.99378881987577 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 91.92546583850931 | 91.97535744486666 | 91.9254658385093 | 1610 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 93.35403726708074 | 93.40872195279442 | 93.35403726708074 | 1610 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 74.34782608695653 | 68.44642268462177 | 74.34782608695653 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 91.86335403726707 | 91.8830483121891 | 91.86335403726709 | 1610 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 92.98136645962732 | 93.0450526766141 | 92.98136645962732 | 1610 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 77.39130434782608 | 73.1937761810188 | 77.3913043478261 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 92.29813664596274 | 92.34362555951309 | 92.29813664596274 | 1610 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 93.47826086956522 | 93.48977120421867 | 93.47826086956522 | 1610 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 83.41614906832298 | 81.85535633393522 | 83.41614906832298 | 1610 | ok |
| limubert_x | 0.055 | 1nn | 128 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 92.79503105590062 | 92.79662242001743 | 92.79503105590062 | 1610 | ok |
| limubert_x | 0.055 | prototype | 128 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 93.91304347826087 | 93.93597793884739 | 93.91304347826087 | 1610 | ok |
| limubert_x | 0.055 | ridge | 128 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 88.4472049689441 | 87.79537381861385 | 88.44720496894409 | 1610 | ok |

### usc_had
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | stream | n_devices | accuracy | f1_macro | balanced_accuracy | n_queries | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halo | 2.203 | halo-classifier | 0 | True | True | phone_hip | 1 | 37.99025890797231 | 35.35930856435965 | 44.165111717421006 | 3901 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_hip | 1 | 35.452448090233275 | 33.68827469485391 | 40.67721849218963 | 3901 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_hip | 1 | 51.29453986157395 | 53.59589286278034 | 53.5445940939806 | 3901 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_hip | 1 | 51.29453986157395 | 53.59589286278034 | 53.5445940939806 | 3901 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_hip | 1 | 50.91002307100744 | 52.69190616208258 | 53.26614776101945 | 3901 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_hip | 1 | 51.371443219687265 | 53.09911434783279 | 56.27256802242127 | 3901 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_hip | 1 | 51.98667008459369 | 53.88030355814825 | 53.95882031852829 | 3901 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_hip | 1 | 54.652653165854915 | 57.10093388351264 | 57.14722476419326 | 3901 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_hip | 1 | 56.011279159189954 | 58.568106300250875 | 58.577535028055294 | 3901 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_hip | 1 | 55.754934632145606 | 57.57421611425225 | 58.242249721888065 | 3901 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_hip | 1 | 53.06331709817995 | 54.33600305261225 | 58.18807977357673 | 3901 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_hip | 1 | 55.754934632145606 | 57.922831139828666 | 57.867211486142445 | 3901 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_hip | 1 | 60.52294283517047 | 62.37569971649284 | 62.421290776123264 | 3901 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_hip | 1 | 61.061266341963595 | 62.87669755364457 | 63.07810467743643 | 3901 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_hip | 1 | 60.75365290951038 | 62.18446063360496 | 63.02485644778897 | 3901 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_hip | 1 | 55.57549346321456 | 56.42820134498866 | 60.626837731539105 | 3901 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_hip | 1 | 61.163804152781346 | 62.81072719823263 | 62.827935459383134 | 3901 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_hip | 1 | 65.1884132273776 | 66.4817456782025 | 66.51999175969028 | 3901 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_hip | 1 | 63.599077159702645 | 64.91640699611555 | 65.29227219564471 | 3901 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_hip | 1 | 64.44501409894899 | 65.30363899628956 | 66.34160813617876 | 3901 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_hip | 1 | 59.087413483722116 | 59.263151744729115 | 63.58207659964881 | 3901 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_hip | 1 | 66.88028710587028 | 68.15083197145012 | 68.09367684829903 | 3901 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_hip | 1 | 70.34093822096898 | 70.89471060945623 | 71.04669797857225 | 3901 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_hip | 1 | 65.23968213278647 | 66.49310385211693 | 67.029103918903 | 3901 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_hip | 1 | 67.88003076134325 | 68.22285326508278 | 69.12250708064533 | 3901 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_hip | 1 | 61.035631889259164 | 61.02664671774293 | 65.21424985027544 | 3901 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_hip | 1 | 70.2127659574468 | 70.90694359587951 | 71.0013744778403 | 3901 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_hip | 1 | 75.77544219430915 | 75.46969656875167 | 75.48358251829332 | 3901 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_hip | 1 | 66.31632914637272 | 67.18812804786593 | 67.67025325437902 | 3901 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_hip | 1 | 70.03332478851576 | 70.2027861354368 | 71.10057816798184 | 3901 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_hip | 1 | 62.16354780825429 | 61.46628511178967 | 66.10903840019516 | 3901 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_hip | 1 | 73.39143809279672 | 73.67468651661059 | 73.7649296153004 | 3901 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_hip | 1 | 79.49243783645218 | 78.98161002358455 | 79.18811195764982 | 3901 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_hip | 1 | 65.82927454498846 | 66.52183262654881 | 67.16806562502887 | 3901 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_hip | 1 | 72.31479107921047 | 71.9501263501833 | 72.90566447383914 | 3901 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_hip | 1 | 62.88131248397847 | 61.875361872662594 | 66.89131574558728 | 3901 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_hip | 1 | 74.67316072801846 | 74.71004446086137 | 74.89594694684573 | 3901 | ok |
| halo | 0.789 | 1nn | 128 | True | True | phone_hip | 1 | 83.2607023840041 | 82.32552137727295 | 82.45861255189796 | 3901 | ok |
| halo | 0.789 | prototype | 128 | True | True | phone_hip | 1 | 66.88028710587028 | 67.50576790789533 | 68.07067080446024 | 3901 | ok |
| halo | 0.789 | ridge | 128 | True | True | phone_hip | 1 | 75.64726993078698 | 75.1583015864941 | 76.01691175826127 | 3901 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | phone_hip | 1 | 63.52217380158933 | 62.342927142735775 | 67.38383063478474 | 3901 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | phone_hip | 1 | 75.39092540374263 | 75.05281502114464 | 75.2615348932677 | 3901 | ok |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_hip | 1 | 28.50551140733145 | 27.3687392729648 | 31.413312963427714 | 3901 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_hip | 1 | 32.83773391438093 | 33.12998527136032 | 33.62572942514459 | 3901 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_hip | 1 | 32.83773391438093 | 33.12998527136032 | 33.62572942514459 | 3901 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_hip | 1 | 33.65803640092284 | 33.52677981171584 | 34.028870447184836 | 3901 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_hip | 1 | 37.47756985388362 | 37.96123548180119 | 38.639608478292025 | 3901 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_hip | 1 | 37.32376313765701 | 37.97402754994681 | 38.52943678729604 | 3901 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_hip | 1 | 38.19533452960779 | 38.55973822920311 | 39.59128931529869 | 3901 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_hip | 1 | 42.96334273263266 | 43.82507419752778 | 44.54721504230218 | 3901 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_hip | 1 | 41.06639323250448 | 41.95035911021714 | 42.893503641209776 | 3901 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_hip | 1 | 42.527557036657264 | 42.51466171334562 | 44.22889821573456 | 3901 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_hip | 1 | 47.11612407075109 | 48.65335754465178 | 48.92024158055969 | 3901 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_hip | 1 | 42.83517046911049 | 43.95527261857586 | 44.823979995751245 | 3901 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_hip | 1 | 45.83440143552935 | 45.86683618523742 | 47.50260900384254 | 3901 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_hip | 1 | 51.08946423993848 | 52.3413815642261 | 53.045965922687564 | 3901 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_hip | 1 | 46.47526275314022 | 47.31352750110411 | 48.55050939357832 | 3901 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_hip | 1 | 50.140989489874386 | 50.18169459135219 | 52.10672473137608 | 3901 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_hip | 1 | 57.677518584978216 | 58.29990691273069 | 58.8059693729293 | 3901 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_hip | 1 | 47.50064086131761 | 48.28044955736412 | 49.44126164635117 | 3901 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_hip | 1 | 53.47346834145091 | 53.6139394182808 | 55.504677967690554 | 3901 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_hip | 1 | 61.29197641630352 | 61.96556791285444 | 62.3736555707615 | 3901 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_hip | 1 | 50.140989489874386 | 50.850845235252336 | 52.39852619937832 | 3901 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_hip | 1 | 57.19046398359395 | 57.83714456464154 | 59.56724229066269 | 3901 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | phone_hip | 1 | 63.77851832863368 | 64.41736592541537 | 64.94056805210198 | 3901 | ok |
| harnet | 4.49 | prototype | 128 | False | False | phone_hip | 1 | 49.013073570879264 | 49.74000434580592 | 51.599755329699434 | 3901 | ok |
| harnet | 4.49 | ridge | 128 | False | False | phone_hip | 1 | 59.85644706485517 | 60.91871777740209 | 62.45607092627964 | 3901 | ok |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_hip | 1 | 30.274288643937453 | 26.540751526810645 | 33.03129132226783 | 3901 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_hip | 1 | 36.24711612407075 | 38.94233885111142 | 37.93953928951101 | 3901 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_hip | 1 | 36.24711612407075 | 38.94233885111142 | 37.93953928951101 | 3901 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_hip | 1 | 37.78518328633684 | 38.76378483779035 | 39.274637853323604 | 3901 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_hip | 1 | 43.322225070494746 | 46.079600250140444 | 45.58948956213664 | 3901 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_hip | 1 | 41.168931043322225 | 43.873005781571564 | 43.50721540851801 | 3901 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_hip | 1 | 41.63035119200205 | 43.03198827999188 | 43.904228040053475 | 3901 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_hip | 1 | 48.602922327608304 | 50.922961700813794 | 50.54239054686116 | 3901 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_hip | 1 | 42.5531914893617 | 45.78861702503714 | 45.82317999315138 | 3901 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_hip | 1 | 43.45039733401692 | 45.6435207709885 | 46.529070918140654 | 3901 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_hip | 1 | 53.9861573955396 | 55.20312158216484 | 55.111986443341124 | 3901 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_hip | 1 | 45.47551909766727 | 48.41432303172131 | 48.73076115312223 | 3901 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_hip | 1 | 48.0133299154063 | 50.008122700043515 | 50.95413873291752 | 3901 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_hip | 1 | 59.98461932837734 | 61.03154245878769 | 61.07798991857889 | 3901 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_hip | 1 | 46.6803383747757 | 49.218825656281325 | 49.77026750175868 | 3901 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_hip | 1 | 52.78133811843118 | 53.79023574875025 | 55.07156598649362 | 3901 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_hip | 1 | 66.16252243014613 | 66.70626954417895 | 66.76527896617625 | 3901 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_hip | 1 | 49.50012817226352 | 51.26160480155184 | 51.87136590555211 | 3901 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_hip | 1 | 57.113560625480645 | 57.22070268987506 | 58.67167654652734 | 3901 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_hip | 1 | 69.57190463983594 | 69.98339908544983 | 70.02512320668782 | 3901 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_hip | 1 | 52.345552422455775 | 53.212253195237466 | 54.03499727902498 | 3901 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_hip | 1 | 58.83106895667778 | 58.693429165210084 | 60.236397628141425 | 3901 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | phone_hip | 1 | 74.46808510638297 | 74.78783389090958 | 74.92648417662751 | 3901 | ok |
| unimts | 68.61 | prototype | 128 | True | False | phone_hip | 1 | 54.08869520635734 | 54.254222489184954 | 55.442270631641165 | 3901 | ok |
| unimts | 68.61 | ridge | 128 | True | False | phone_hip | 1 | 63.240194821840554 | 62.29842361599987 | 64.00408342006571 | 3901 | ok |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_hip | 1 | 9.10023071007434 | 1.4498439121818079 | 8.342408629731592 | 3901 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_hip | 1 | 16.94437323763138 | 16.20326981394304 | 16.268532344081184 | 3901 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_hip | 1 | 16.94437323763138 | 16.20326981394304 | 16.268532344081184 | 3901 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_hip | 1 | 15.17559600102538 | 14.552096321562852 | 14.93887615536659 | 3901 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_hip | 1 | 20.19994873109459 | 19.491657291362053 | 19.670398659022116 | 3901 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_hip | 1 | 20.45629325813894 | 19.912203238910806 | 20.258505525294126 | 3901 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_hip | 1 | 18.02102025121764 | 17.020003106835137 | 17.49622614066859 | 3901 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_hip | 1 | 22.763394001538067 | 21.74743066697652 | 21.741098377029296 | 3901 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_hip | 1 | 19.635990771597026 | 18.711118589843515 | 18.73431384914751 | 3901 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_hip | 1 | 19.405280697257112 | 17.66750699080135 | 18.16189777758499 | 3901 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_hip | 1 | 25.557549346321455 | 24.74399279202276 | 24.762645141353936 | 3901 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_hip | 1 | 21.27659574468085 | 20.172223651237 | 20.174626513897078 | 3901 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_hip | 1 | 22.14816713663163 | 19.353226892294032 | 20.159908799552706 | 3901 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_hip | 1 | 26.42912073827224 | 25.954372016337146 | 26.038809139117454 | 3901 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_hip | 1 | 21.891822609587287 | 20.969720192782596 | 21.121989569288544 | 3901 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_hip | 1 | 24.12201999487311 | 20.28228872844102 | 21.542040687444594 | 3901 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_hip | 1 | 29.81286849525763 | 28.875279255572668 | 28.950225123354173 | 3901 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_hip | 1 | 21.63547808254294 | 20.235023537737998 | 20.743796251960926 | 3901 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_hip | 1 | 27.428864393745194 | 22.120943681449333 | 24.137519867088773 | 3901 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_hip | 1 | 32.35067931299667 | 31.674673593271546 | 31.75865501603518 | 3901 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_hip | 1 | 22.68649064342476 | 20.822852735650432 | 21.671791513202603 | 3901 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_hip | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | 1nn | 128 | True | False | phone_hip | 1 | 36.19584721866188 | 35.3197068669706 | 35.45606647625854 | 3901 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | phone_hip | 1 | 23.173545244809024 | 20.871119149255684 | 22.145954486625293 | 3901 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | phone_hip | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_hip | 1 | 13.406818764419379 | 12.602363853000067 | 15.236372021695802 | 3901 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_hip | 1 | 47.11612407075109 | 45.35757290191955 | 45.08842767221052 | 3901 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_hip | 1 | 47.11612407075109 | 45.35757290191955 | 45.08842767221052 | 3901 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_hip | 1 | 44.06562419892335 | 41.69160212811911 | 44.04124614134952 | 3901 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_hip | 1 | 54.908997692899256 | 52.60075561727694 | 52.22669178028861 | 3901 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_hip | 1 | 54.319405280697254 | 51.56978024895093 | 51.2465383240419 | 3901 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_hip | 1 | 49.06434247628813 | 45.570010485290815 | 48.37627157187615 | 3901 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_hip | 1 | 63.650346065111506 | 61.14651551100907 | 60.53747983130303 | 3901 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_hip | 1 | 58.83106895667778 | 56.21939189723898 | 55.61027661100938 | 3901 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_hip | 1 | 54.883363240194825 | 50.867627612461455 | 53.73463870940118 | 3901 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_hip | 1 | 69.11048449115611 | 66.46674085334425 | 65.85308879515017 | 3901 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_hip | 1 | 62.368623429889766 | 59.14855203235236 | 58.58605703408822 | 3901 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_hip | 1 | 58.677262240451164 | 54.44025638891924 | 57.20670469154903 | 3901 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_hip | 1 | 74.80133299154062 | 72.09139697373922 | 71.56474895867414 | 3901 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_hip | 1 | 64.82953088951551 | 61.58433374985955 | 61.04755320078847 | 3901 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_hip | 1 | 60.67674955139708 | 56.09733046716019 | 59.092734656047554 | 3901 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_hip | 1 | 80.05639579594975 | 77.65494763892545 | 77.16783596961417 | 3901 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_hip | 1 | 65.98308126121508 | 62.89600647672776 | 62.42962521045499 | 3901 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_hip | 1 | 62.573699051525246 | 58.23645828202705 | 60.998716464780934 | 3901 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_hip | 1 | 83.23506793129967 | 80.78855203290115 | 80.36357175049976 | 3901 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_hip | 1 | 67.34170725455012 | 64.31899299051219 | 63.83863287686141 | 3901 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_hip | 1 | 65.8549089976929 | 62.091019220761844 | 64.36845817309599 | 3901 | ok |
| limubert_x | 0.055 | 1nn | 128 | False | False | phone_hip | 1 | 84.51679056652141 | 82.03096800371372 | 81.66485776506406 | 3901 | ok |
| limubert_x | 0.055 | prototype | 128 | False | False | phone_hip | 1 | 67.59805178159446 | 64.5948901493459 | 64.02472159811349 | 3901 | ok |
| limubert_x | 0.055 | ridge | 128 | False | False | phone_hip | 1 | 67.44424506536785 | 63.95518019684639 | 65.79700061614894 | 3901 | ok |

### ut_complex
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | stream | n_devices | accuracy | f1_macro | balanced_accuracy | n_queries | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halo | 2.203 | halo-classifier | 0 | True | True | watch_wrist | 1 | 45.719063545150505 | 40.77583445302061 | 45.719063545150505 | 2990 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | watch_wrist | 1 | 28.127090301003342 | 25.998954753716795 | 28.12709030100335 | 2990 | ok |
| halo | 0.789 | 1nn | 1 | True | True | watch_wrist | 1 | 64.44816053511705 | 64.16868996072958 | 64.44816053511705 | 2990 | ok |
| halo | 0.789 | prototype | 1 | True | True | watch_wrist | 1 | 64.44816053511705 | 64.16868996072958 | 64.44816053511705 | 2990 | ok |
| halo | 0.789 | ridge | 1 | True | True | watch_wrist | 1 | 64.38127090301003 | 63.99834998919567 | 64.38127090301003 | 2990 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | watch_wrist | 1 | 66.08695652173913 | 65.48810284219363 | 66.08695652173913 | 2990 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | watch_wrist | 1 | 64.2809364548495 | 63.850524018369235 | 64.2809364548495 | 2990 | ok |
| halo | 0.789 | 1nn | 2 | True | True | watch_wrist | 1 | 71.43812709030101 | 71.13685807767187 | 71.43812709030102 | 2990 | ok |
| halo | 0.789 | prototype | 2 | True | True | watch_wrist | 1 | 73.07692307692307 | 72.8006140225336 | 73.07692307692307 | 2990 | ok |
| halo | 0.789 | ridge | 2 | True | True | watch_wrist | 1 | 73.24414715719062 | 72.67872011850397 | 73.24414715719062 | 2990 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | watch_wrist | 1 | 72.07357859531773 | 71.41963750290792 | 72.07357859531773 | 2990 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | watch_wrist | 1 | 72.60869565217392 | 72.1933381883659 | 72.60869565217392 | 2990 | ok |
| halo | 0.789 | 1nn | 4 | True | True | watch_wrist | 1 | 74.78260869565217 | 74.61275295429427 | 74.78260869565217 | 2990 | ok |
| halo | 0.789 | prototype | 4 | True | True | watch_wrist | 1 | 77.29096989966555 | 77.15626260188905 | 77.29096989966554 | 2990 | ok |
| halo | 0.789 | ridge | 4 | True | True | watch_wrist | 1 | 77.9933110367893 | 77.50473730293236 | 77.99331103678931 | 2990 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | watch_wrist | 1 | 75.75250836120402 | 75.30588921769517 | 75.75250836120401 | 2990 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | watch_wrist | 1 | 76.12040133779264 | 75.80865222706466 | 76.12040133779264 | 2990 | ok |
| halo | 0.789 | 1nn | 8 | True | True | watch_wrist | 1 | 77.35785953177258 | 77.24041235453959 | 77.35785953177258 | 2990 | ok |
| halo | 0.789 | prototype | 8 | True | True | watch_wrist | 1 | 79.4314381270903 | 79.25540205266721 | 79.43143812709029 | 2990 | ok |
| halo | 0.789 | ridge | 8 | True | True | watch_wrist | 1 | 79.93311036789298 | 79.4288887047396 | 79.93311036789297 | 2990 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | watch_wrist | 1 | 77.05685618729096 | 76.66151929851324 | 77.05685618729096 | 2990 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | watch_wrist | 1 | 78.76254180602007 | 78.39173579917032 | 78.76254180602007 | 2990 | ok |
| halo | 0.789 | 1nn | 16 | True | True | watch_wrist | 1 | 80.5351170568562 | 80.42143794390977 | 80.5351170568562 | 2990 | ok |
| halo | 0.789 | prototype | 16 | True | True | watch_wrist | 1 | 81.77257525083613 | 81.69573544479003 | 81.7725752508361 | 2990 | ok |
| halo | 0.789 | ridge | 16 | True | True | watch_wrist | 1 | 83.51170568561874 | 83.1824587361501 | 83.51170568561872 | 2990 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | watch_wrist | 1 | 79.49832775919732 | 79.22821874038142 | 79.49832775919732 | 2990 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | watch_wrist | 1 | 82.6086956521739 | 82.26728852962492 | 82.60869565217394 | 2990 | ok |
| halo | 0.789 | 1nn | 32 | True | True | watch_wrist | 1 | 82.10702341137124 | 82.00150345833805 | 82.10702341137124 | 2990 | ok |
| halo | 0.789 | prototype | 32 | True | True | watch_wrist | 1 | 82.3076923076923 | 82.24399352838638 | 82.3076923076923 | 2990 | ok |
| halo | 0.789 | ridge | 32 | True | True | watch_wrist | 1 | 84.58193979933111 | 84.30033533630119 | 84.58193979933107 | 2990 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | watch_wrist | 1 | 79.63210702341136 | 79.35649946045693 | 79.63210702341138 | 2990 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | watch_wrist | 1 | 84.11371237458194 | 83.8630855800977 | 84.11371237458195 | 2990 | ok |
| halo | 0.789 | 1nn | 64 | True | True | watch_wrist | 1 | 82.2742474916388 | 82.23550968302766 | 82.2742474916388 | 2990 | ok |
| halo | 0.789 | prototype | 64 | True | True | watch_wrist | 1 | 82.5752508361204 | 82.54020022709108 | 82.57525083612039 | 2990 | ok |
| halo | 0.789 | ridge | 64 | True | True | watch_wrist | 1 | 85.51839464882943 | 85.31966008825195 | 85.51839464882943 | 2990 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | watch_wrist | 1 | 79.76588628762542 | 79.43195852506857 | 79.76588628762542 | 2990 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | watch_wrist | 1 | 84.71571906354515 | 84.43460384393464 | 84.71571906354515 | 2990 | ok |
| halo | 0.789 | 1nn | 128 | True | True | watch_wrist | 1 | 83.01003344481606 | 82.97587854041019 | 83.01003344481607 | 2990 | ok |
| halo | 0.789 | prototype | 128 | True | True | watch_wrist | 1 | 82.3076923076923 | 82.24991084622718 | 82.30769230769232 | 2990 | ok |
| halo | 0.789 | ridge | 128 | True | True | watch_wrist | 1 | 86.38795986622073 | 86.19539470708747 | 86.38795986622074 | 2990 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | watch_wrist | 1 | 80.10033444816054 | 79.83701203931139 | 80.10033444816051 | 2990 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | watch_wrist | 1 | 84.94983277591973 | 84.68498896226252 | 84.94983277591973 | 2990 | ok |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | watch_wrist | 1 | 36.65551839464883 | 33.63956183984052 | 36.65551839464883 | 2990 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | watch_wrist | 1 | 48.16053511705686 | 47.484268902585974 | 48.16053511705685 | 2990 | ok |
| harnet | 4.49 | prototype | 1 | False | False | watch_wrist | 1 | 48.16053511705686 | 47.484268902585974 | 48.16053511705685 | 2990 | ok |
| harnet | 4.49 | ridge | 1 | False | False | watch_wrist | 1 | 48.22742474916388 | 47.148799217222944 | 48.22742474916387 | 2990 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | watch_wrist | 1 | 54.04682274247492 | 53.48586135606088 | 54.04682274247492 | 2990 | ok |
| harnet | 4.49 | prototype | 2 | False | False | watch_wrist | 1 | 54.515050167224075 | 53.81042409142931 | 54.515050167224075 | 2990 | ok |
| harnet | 4.49 | ridge | 2 | False | False | watch_wrist | 1 | 54.147157190635454 | 53.01871942068033 | 54.147157190635454 | 2990 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | watch_wrist | 1 | 59.49832775919732 | 59.24782583318735 | 59.49832775919733 | 2990 | ok |
| harnet | 4.49 | prototype | 4 | False | False | watch_wrist | 1 | 58.16053511705685 | 57.71203662692628 | 58.16053511705685 | 2990 | ok |
| harnet | 4.49 | ridge | 4 | False | False | watch_wrist | 1 | 60.735785953177256 | 59.54824945750858 | 60.735785953177256 | 2990 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | watch_wrist | 1 | 63.21070234113713 | 63.20197052512031 | 63.210702341137136 | 2990 | ok |
| harnet | 4.49 | prototype | 8 | False | False | watch_wrist | 1 | 62.17391304347826 | 61.86321430002203 | 62.173913043478265 | 2990 | ok |
| harnet | 4.49 | ridge | 8 | False | False | watch_wrist | 1 | 66.82274247491638 | 65.93487874102783 | 66.82274247491638 | 2990 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | watch_wrist | 1 | 66.98996655518394 | 66.95940041326925 | 66.98996655518395 | 2990 | ok |
| harnet | 4.49 | prototype | 16 | False | False | watch_wrist | 1 | 65.15050167224081 | 64.98399303459428 | 65.15050167224081 | 2990 | ok |
| harnet | 4.49 | ridge | 16 | False | False | watch_wrist | 1 | 73.17725752508362 | 72.42942880777458 | 73.1772575250836 | 2990 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | watch_wrist | 1 | 69.1304347826087 | 69.13211894259199 | 69.1304347826087 | 2990 | ok |
| harnet | 4.49 | prototype | 32 | False | False | watch_wrist | 1 | 66.05351170568562 | 65.80697300951482 | 66.05351170568561 | 2990 | ok |
| harnet | 4.49 | ridge | 32 | False | False | watch_wrist | 1 | 76.12040133779264 | 75.392518655207 | 76.12040133779263 | 2990 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | watch_wrist | 1 | 70.93645484949833 | 71.04578005935659 | 70.93645484949833 | 2990 | ok |
| harnet | 4.49 | prototype | 64 | False | False | watch_wrist | 1 | 67.62541806020067 | 67.40884911553862 | 67.62541806020066 | 2990 | ok |
| harnet | 4.49 | ridge | 64 | False | False | watch_wrist | 1 | 79.86622073578596 | 79.25584334686805 | 79.86622073578596 | 2990 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | watch_wrist | 1 | 72.47491638795987 | 72.64646229692906 | 72.47491638795987 | 2990 | ok |
| harnet | 4.49 | prototype | 128 | False | False | watch_wrist | 1 | 68.16053511705685 | 67.94864125523516 | 68.16053511705687 | 2990 | ok |
| harnet | 4.49 | ridge | 128 | False | False | watch_wrist | 1 | 81.20401337792642 | 80.74346068267607 | 81.20401337792642 | 2990 | ok |
| unimts | 68.61 | native_zero_support | 0 | True | False | watch_wrist | 1 | 28.19397993311037 | 21.824378692345164 | 28.193979933110363 | 2990 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | watch_wrist | 1 | 50.468227424749166 | 50.282860416764976 | 50.468227424749166 | 2990 | ok |
| unimts | 68.61 | prototype | 1 | True | False | watch_wrist | 1 | 50.468227424749166 | 50.282860416764976 | 50.468227424749166 | 2990 | ok |
| unimts | 68.61 | ridge | 1 | True | False | watch_wrist | 1 | 48.42809364548495 | 47.77567640956304 | 48.42809364548496 | 2990 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | watch_wrist | 1 | 55.01672240802675 | 55.018568572547885 | 55.01672240802675 | 2990 | ok |
| unimts | 68.61 | prototype | 2 | True | False | watch_wrist | 1 | 55.88628762541806 | 55.76191345193835 | 55.88628762541806 | 2990 | ok |
| unimts | 68.61 | ridge | 2 | True | False | watch_wrist | 1 | 52.44147157190635 | 51.68792321651539 | 52.44147157190635 | 2990 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | watch_wrist | 1 | 61.337792642140464 | 61.380759626523606 | 61.33779264214046 | 2990 | ok |
| unimts | 68.61 | prototype | 4 | True | False | watch_wrist | 1 | 60.40133779264214 | 60.29936614176059 | 60.40133779264214 | 2990 | ok |
| unimts | 68.61 | ridge | 4 | True | False | watch_wrist | 1 | 58.42809364548495 | 57.6237273169829 | 58.42809364548496 | 2990 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | watch_wrist | 1 | 65.38461538461539 | 65.4585938772262 | 65.38461538461539 | 2990 | ok |
| unimts | 68.61 | prototype | 8 | True | False | watch_wrist | 1 | 62.909698996655514 | 62.77991903742136 | 62.90969899665553 | 2990 | ok |
| unimts | 68.61 | ridge | 8 | True | False | watch_wrist | 1 | 62.408026755852845 | 61.58097205308069 | 62.40802675585285 | 2990 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | watch_wrist | 1 | 67.45819397993311 | 67.54384630080682 | 67.45819397993313 | 2990 | ok |
| unimts | 68.61 | prototype | 16 | True | False | watch_wrist | 1 | 64.68227424749163 | 64.61297857265144 | 64.68227424749163 | 2990 | ok |
| unimts | 68.61 | ridge | 16 | True | False | watch_wrist | 1 | 65.45150501672241 | 64.6004639918307 | 65.4515050167224 | 2990 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | watch_wrist | 1 | 68.72909698996655 | 68.93570608116595 | 68.72909698996655 | 2990 | ok |
| unimts | 68.61 | prototype | 32 | True | False | watch_wrist | 1 | 65.61872909698997 | 65.41831773637642 | 65.61872909698997 | 2990 | ok |
| unimts | 68.61 | ridge | 32 | True | False | watch_wrist | 1 | 67.82608695652173 | 66.83584991209952 | 67.82608695652173 | 2990 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | watch_wrist | 1 | 70.96989966555184 | 71.2460417096067 | 70.96989966555184 | 2990 | ok |
| unimts | 68.61 | prototype | 64 | True | False | watch_wrist | 1 | 66.38795986622074 | 66.13249819822033 | 66.38795986622074 | 2990 | ok |
| unimts | 68.61 | ridge | 64 | True | False | watch_wrist | 1 | 71.1371237458194 | 70.4392769295464 | 71.1371237458194 | 2990 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | watch_wrist | 1 | 72.97658862876254 | 73.127912223539 | 72.97658862876256 | 2990 | ok |
| unimts | 68.61 | prototype | 128 | True | False | watch_wrist | 1 | 67.25752508361204 | 66.9060026076796 | 67.25752508361205 | 2990 | ok |
| unimts | 68.61 | ridge | 128 | True | False | watch_wrist | 1 | 74.44816053511705 | 74.04244595875018 | 74.44816053511704 | 2990 | ok |
| normwear | 1293.86 | native_zero_support | 0 | True | False | watch_wrist | 1 | 7.6923076923076925 | 1.0989010989010988 | 7.6923076923076925 | 2990 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | watch_wrist | 1 | 23.745819397993312 | 23.529761360557934 | 23.745819397993312 | 2990 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | watch_wrist | 1 | 23.745819397993312 | 23.529761360557934 | 23.745819397993312 | 2990 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | watch_wrist | 1 | 16.82274247491639 | 15.539723098776111 | 16.82274247491639 | 2990 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | watch_wrist | 1 | 26.45484949832776 | 26.364494150883015 | 26.45484949832776 | 2990 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | watch_wrist | 1 | 25.65217391304348 | 25.280510867463175 | 25.65217391304348 | 2990 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | watch_wrist | 1 | 17.759197324414718 | 14.893359742482836 | 17.759197324414714 | 2990 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | watch_wrist | 1 | 29.899665551839465 | 29.696250552009236 | 29.899665551839462 | 2990 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | watch_wrist | 1 | 28.327759197324415 | 27.448584685645283 | 28.327759197324415 | 2990 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | watch_wrist | 1 | 21.20401337792642 | 17.520848278117366 | 21.204013377926422 | 2990 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | watch_wrist | 1 | 33.47826086956522 | 33.52318097771082 | 33.47826086956522 | 2990 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | watch_wrist | 1 | 31.371237458193978 | 29.965583083145596 | 31.371237458193978 | 2990 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | watch_wrist | 1 | 23.612040133779264 | 18.340451597230537 | 23.61204013377927 | 2990 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | watch_wrist | 1 | 36.75585284280937 | 36.74046543770125 | 36.75585284280937 | 2990 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | watch_wrist | 1 | 32.441471571906355 | 30.49630732213978 | 32.441471571906355 | 2990 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | watch_wrist | 1 | 27.692307692307693 | 19.9393751684251 | 27.692307692307693 | 2990 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | watch_wrist | 1 | 39.364548494983275 | 39.34390628980286 | 39.364548494983275 | 2990 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | watch_wrist | 1 | 32.474916387959865 | 29.970973344798 | 32.474916387959865 | 2990 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | watch_wrist | 1 | 31.83946488294314 | 22.065570958677718 | 31.839464882943147 | 2990 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | watch_wrist | 1 | 43.54515050167224 | 43.55641863177947 | 43.545150501672246 | 2990 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | watch_wrist | 1 | 32.30769230769231 | 29.822695455919785 | 32.30769230769231 | 2990 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | watch_wrist | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | 1nn | 128 | True | False | watch_wrist | 1 | 44.91638795986622 | 44.86179834269036 | 44.91638795986622 | 2990 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | watch_wrist | 1 | 32.97658862876254 | 30.374545982681173 | 32.97658862876255 | 2990 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | watch_wrist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | watch_wrist | 1 | 10.334448160535118 | 9.549137251978635 | 10.334448160535116 | 2990 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | watch_wrist | 1 | 54.61538461538461 | 53.65072581761991 | 54.61538461538461 | 2990 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | watch_wrist | 1 | 54.61538461538461 | 53.65072581761991 | 54.61538461538461 | 2990 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | watch_wrist | 1 | 45.01672240802675 | 40.84586144975161 | 45.01672240802675 | 2990 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | watch_wrist | 1 | 60.802675585284284 | 60.36451442468882 | 60.80267558528429 | 2990 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | watch_wrist | 1 | 58.32775919732441 | 57.23393825411514 | 58.327759197324426 | 2990 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | watch_wrist | 1 | 47.525083612040135 | 42.37637396478273 | 47.525083612040135 | 2990 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | watch_wrist | 1 | 66.62207357859532 | 66.14215596519239 | 66.62207357859529 | 2990 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | watch_wrist | 1 | 63.177257525083604 | 62.02824593307624 | 63.17725752508362 | 2990 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | watch_wrist | 1 | 51.438127090301 | 45.7903641705714 | 51.43812709030101 | 2990 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | watch_wrist | 1 | 71.90635451505017 | 71.61744799757935 | 71.90635451505017 | 2990 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | watch_wrist | 1 | 65.01672240802677 | 63.78591026386926 | 65.01672240802677 | 2990 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | watch_wrist | 1 | 54.91638795986622 | 49.48522562770259 | 54.91638795986623 | 2990 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | watch_wrist | 1 | 75.91973244147158 | 75.96000451250146 | 75.91973244147158 | 2990 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | watch_wrist | 1 | 66.95652173913044 | 65.68780328059613 | 66.95652173913044 | 2990 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | watch_wrist | 1 | 59.063545150501675 | 53.38397319892334 | 59.06354515050166 | 2990 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | watch_wrist | 1 | 78.32775919732441 | 78.22248548062323 | 78.32775919732443 | 2990 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | watch_wrist | 1 | 68.42809364548495 | 66.95483125795961 | 68.42809364548495 | 2990 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | watch_wrist | 1 | 63.64548494983278 | 58.70074006757198 | 63.64548494983278 | 2990 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | watch_wrist | 1 | 79.6989966555184 | 79.64512898977405 | 79.69899665551839 | 2990 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | watch_wrist | 1 | 68.96321070234114 | 67.43427350889549 | 68.96321070234114 | 2990 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | watch_wrist | 1 | 67.4247491638796 | 63.543928863632814 | 67.4247491638796 | 2990 | ok |
| limubert_x | 0.055 | 1nn | 128 | False | False | watch_wrist | 1 | 81.80602006688963 | 81.74681578497471 | 81.80602006688964 | 2990 | ok |
| limubert_x | 0.055 | prototype | 128 | False | False | watch_wrist | 1 | 68.62876254180603 | 66.96642065729203 | 68.62876254180603 | 2990 | ok |
| limubert_x | 0.055 | ridge | 128 | False | False | watch_wrist | 1 | 71.87290969899665 | 68.70411081845987 | 71.87290969899665 | 2990 | ok |

### Dataset-Balanced Mean
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | accuracy | f1_macro | n_datasets |
|---|---|---|---|---|---|---|---|---|
| halo | 0.789 | 1nn | 1 | True | True | 59.97 | 59.397 | 6 |
| halo | 0.789 | 1nn | 2 | True | True | 64.319 | 63.976 | 6 |
| halo | 0.789 | 1nn | 4 | True | True | 68.111 | 67.905 | 6 |
| halo | 0.789 | 1nn | 8 | True | True | 70.636 | 70.461 | 6 |
| halo | 0.789 | 1nn | 16 | True | True | 72.604 | 72.269 | 6 |
| halo | 0.789 | 1nn | 32 | True | True | 74.46 | 73.685 | 6 |
| halo | 0.789 | 1nn | 64 | True | True | 75.71 | 75.42 | 6 |
| halo | 0.789 | 1nn | 128 | True | True | 77.317 | 77.265 | 6 |
| halo | 2.203 | halo-classifier | 0 | True | True | 53.828 | 50.197 | 6 |
| halo | 2.203 | halo-classifier | 1 | True | True | 62.015 | 61.054 | 6 |
| halo | 2.203 | halo-classifier | 2 | True | True | 65.362 | 64.525 | 6 |
| halo | 2.203 | halo-classifier | 4 | True | True | 67.583 | 66.77 | 6 |
| halo | 2.203 | halo-classifier | 8 | True | True | 69.757 | 68.948 | 6 |
| halo | 2.203 | halo-classifier | 16 | True | True | 70.87 | 70.068 | 6 |
| halo | 2.203 | halo-classifier | 32 | True | True | 71.148 | 69.77 | 6 |
| halo | 2.203 | halo-classifier | 64 | True | True | 71.735 | 70.209 | 6 |
| halo | 2.203 | halo-classifier | 128 | True | True | 72.906 | 71.503 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | 59.859 | 59.073 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | 64.996 | 64.495 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | 69.281 | 68.864 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | 72.422 | 72.135 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | 74.392 | 73.962 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | 76.084 | 75.283 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | 76.863 | 76.487 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | 77.854 | 77.693 | 6 |
| halo | 0.789 | prototype | 1 | True | True | 59.97 | 59.397 | 6 |
| halo | 0.789 | prototype | 2 | True | True | 65.188 | 64.585 | 6 |
| halo | 0.789 | prototype | 4 | True | True | 69.674 | 69.096 | 6 |
| halo | 0.789 | prototype | 8 | True | True | 71.708 | 71.123 | 6 |
| halo | 0.789 | prototype | 16 | True | True | 73.435 | 72.919 | 6 |
| halo | 0.789 | prototype | 32 | True | True | 74.15 | 73.507 | 6 |
| halo | 0.789 | prototype | 64 | True | True | 74.629 | 74.051 | 6 |
| halo | 0.789 | prototype | 128 | True | True | 75.627 | 75.105 | 6 |
| halo | 0.789 | ridge | 1 | True | True | 60.091 | 59.243 | 6 |
| halo | 0.789 | ridge | 2 | True | True | 65.633 | 64.728 | 6 |
| halo | 0.789 | ridge | 4 | True | True | 70.22 | 69.442 | 6 |
| halo | 0.789 | ridge | 8 | True | True | 72.653 | 72.054 | 6 |
| halo | 0.789 | ridge | 16 | True | True | 75.249 | 74.681 | 6 |
| halo | 0.789 | ridge | 32 | True | True | 76.536 | 75.978 | 6 |
| halo | 0.789 | ridge | 64 | True | True | 77.45 | 76.835 | 6 |
| halo | 0.789 | ridge | 128 | True | True | 79.178 | 78.647 | 6 |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | 43.079 | 38.323 | 6 |
| harnet | 4.49 | 1nn | 1 | False | False | 46.157 | 45.068 | 6 |
| harnet | 4.49 | 1nn | 2 | False | False | 50.308 | 49.619 | 6 |
| harnet | 4.49 | 1nn | 4 | False | False | 54.041 | 53.684 | 6 |
| harnet | 4.49 | 1nn | 8 | False | False | 57.101 | 57.018 | 6 |
| harnet | 4.49 | 1nn | 16 | False | False | 59.767 | 59.827 | 6 |
| harnet | 4.49 | 1nn | 32 | False | False | 62.498 | 62.26 | 6 |
| harnet | 4.49 | 1nn | 64 | False | False | 64.52 | 64.529 | 6 |
| harnet | 4.49 | 1nn | 128 | False | False | 66.176 | 66.394 | 6 |
| harnet | 4.49 | prototype | 1 | False | False | 46.157 | 45.068 | 6 |
| harnet | 4.49 | prototype | 2 | False | False | 50.215 | 49.343 | 6 |
| harnet | 4.49 | prototype | 4 | False | False | 53.635 | 53.074 | 6 |
| harnet | 4.49 | prototype | 8 | False | False | 56.006 | 55.604 | 6 |
| harnet | 4.49 | prototype | 16 | False | False | 58.559 | 58.251 | 6 |
| harnet | 4.49 | prototype | 32 | False | False | 59.499 | 58.953 | 6 |
| harnet | 4.49 | prototype | 64 | False | False | 60.892 | 60.606 | 6 |
| harnet | 4.49 | prototype | 128 | False | False | 60.83 | 60.525 | 6 |
| harnet | 4.49 | ridge | 1 | False | False | 46.047 | 44.621 | 6 |
| harnet | 4.49 | ridge | 2 | False | False | 50.287 | 49.086 | 6 |
| harnet | 4.49 | ridge | 4 | False | False | 54.85 | 53.858 | 6 |
| harnet | 4.49 | ridge | 8 | False | False | 58.886 | 58.272 | 6 |
| harnet | 4.49 | ridge | 16 | False | False | 62.746 | 62.483 | 6 |
| harnet | 4.49 | ridge | 32 | False | False | 65.294 | 64.696 | 6 |
| harnet | 4.49 | ridge | 64 | False | False | 68.327 | 68.139 | 6 |
| harnet | 4.49 | ridge | 128 | False | False | 70.564 | 70.592 | 6 |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | 39.465 | 35.589 | 6 |
| limubert_x | 0.055 | 1nn | 1 | False | False | 50.629 | 49.674 | 5 |
| limubert_x | 0.055 | 1nn | 2 | False | False | 57.527 | 56.509 | 5 |
| limubert_x | 0.055 | 1nn | 4 | False | False | 62.804 | 61.751 | 5 |
| limubert_x | 0.055 | 1nn | 8 | False | False | 67.215 | 66.096 | 5 |
| limubert_x | 0.055 | 1nn | 16 | False | False | 70.842 | 70.005 | 5 |
| limubert_x | 0.055 | 1nn | 32 | False | False | 72.357 | 71.617 | 5 |
| limubert_x | 0.055 | 1nn | 64 | False | False | 73.839 | 73.141 | 5 |
| limubert_x | 0.055 | 1nn | 128 | False | False | 74.292 | 73.703 | 5 |
| limubert_x | 0.055 | prototype | 1 | False | False | 50.629 | 49.674 | 5 |
| limubert_x | 0.055 | prototype | 2 | False | False | 55.086 | 53.886 | 5 |
| limubert_x | 0.055 | prototype | 4 | False | False | 58.392 | 57.225 | 5 |
| limubert_x | 0.055 | prototype | 8 | False | False | 62.049 | 60.691 | 5 |
| limubert_x | 0.055 | prototype | 16 | False | False | 63.897 | 62.362 | 5 |
| limubert_x | 0.055 | prototype | 32 | False | False | 64.986 | 63.314 | 5 |
| limubert_x | 0.055 | prototype | 64 | False | False | 65.724 | 63.839 | 5 |
| limubert_x | 0.055 | prototype | 128 | False | False | 65.557 | 63.395 | 5 |
| limubert_x | 0.055 | ridge | 1 | False | False | 45.931 | 43.036 | 5 |
| limubert_x | 0.055 | ridge | 2 | False | False | 48.714 | 45.248 | 5 |
| limubert_x | 0.055 | ridge | 4 | False | False | 51.938 | 48.287 | 5 |
| limubert_x | 0.055 | ridge | 8 | False | False | 56.646 | 52.745 | 5 |
| limubert_x | 0.055 | ridge | 16 | False | False | 59.923 | 55.471 | 5 |
| limubert_x | 0.055 | ridge | 32 | False | False | 62.905 | 58.272 | 5 |
| limubert_x | 0.055 | ridge | 64 | False | False | 65.209 | 60.9 | 5 |
| limubert_x | 0.055 | ridge | 128 | False | False | 67.18 | 63.185 | 5 |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | 28.012 | 24.441 | 5 |
| normwear | 1293.86 | 1nn | 1 | True | False | 22.035 | 21.475 | 6 |
| normwear | 1293.86 | 1nn | 2 | True | False | 24.323 | 23.738 | 6 |
| normwear | 1293.86 | 1nn | 4 | True | False | 27.241 | 26.544 | 6 |
| normwear | 1293.86 | 1nn | 8 | True | False | 30.369 | 29.752 | 6 |
| normwear | 1293.86 | 1nn | 16 | True | False | 33.134 | 32.608 | 6 |
| normwear | 1293.86 | 1nn | 32 | True | False | 36.086 | 35.314 | 6 |
| normwear | 1293.86 | 1nn | 64 | True | False | 38.957 | 38.377 | 6 |
| normwear | 1293.86 | 1nn | 128 | True | False | 41.056 | 40.604 | 6 |
| normwear | 1293.86 | native_zero_support | 0 | True | False | 14.567 | 3.449 | 6 |
| normwear | 1293.86 | prototype | 1 | True | False | 22.035 | 21.475 | 6 |
| normwear | 1293.86 | prototype | 2 | True | False | 23.779 | 23.128 | 6 |
| normwear | 1293.86 | prototype | 4 | True | False | 25.514 | 24.512 | 6 |
| normwear | 1293.86 | prototype | 8 | True | False | 26.573 | 25.233 | 6 |
| normwear | 1293.86 | prototype | 16 | True | False | 27.639 | 26.138 | 6 |
| normwear | 1293.86 | prototype | 32 | True | False | 27.817 | 25.972 | 6 |
| normwear | 1293.86 | prototype | 64 | True | False | 27.942 | 25.818 | 6 |
| normwear | 1293.86 | prototype | 128 | True | False | 28.272 | 25.91 | 6 |
| normwear | 1293.86 | ridge | 1 | True | False | 19.835 | 18.92 | 6 |
| normwear | 1293.86 | ridge | 2 | True | False | 21.251 | 19.613 | 6 |
| normwear | 1293.86 | ridge | 4 | True | False | 23.417 | 21.002 | 6 |
| normwear | 1293.86 | ridge | 8 | True | False | 25.116 | 21.735 | 6 |
| normwear | 1293.86 | ridge | 16 | True | False | 26.88 | 22.421 | 6 |
| normwear | 1293.86 | ridge | 32 | True | False | 29.279 | 23.587 | 6 |
| normwear | 1293.86 | ridge | 64 | True | False | 30.691 | 25.983 | 4 |
| unimts | 68.61 | 1nn | 1 | True | False | 52.303 | 52.648 | 6 |
| unimts | 68.61 | 1nn | 2 | True | False | 57.891 | 58.465 | 6 |
| unimts | 68.61 | 1nn | 4 | True | False | 62.712 | 63.347 | 6 |
| unimts | 68.61 | 1nn | 8 | True | False | 65.815 | 66.295 | 6 |
| unimts | 68.61 | 1nn | 16 | True | False | 68.421 | 68.879 | 6 |
| unimts | 68.61 | 1nn | 32 | True | False | 70.412 | 70.365 | 6 |
| unimts | 68.61 | 1nn | 64 | True | False | 72.46 | 73.141 | 6 |
| unimts | 68.61 | 1nn | 128 | True | False | 74.81 | 75.195 | 6 |
| unimts | 68.61 | native_zero_support | 0 | True | False | 38.202 | 30.695 | 6 |
| unimts | 68.61 | prototype | 1 | True | False | 52.303 | 52.648 | 6 |
| unimts | 68.61 | prototype | 2 | True | False | 55.477 | 56.109 | 6 |
| unimts | 68.61 | prototype | 4 | True | False | 58.611 | 59.731 | 6 |
| unimts | 68.61 | prototype | 8 | True | False | 60.405 | 61.535 | 6 |
| unimts | 68.61 | prototype | 16 | True | False | 62.535 | 63.583 | 6 |
| unimts | 68.61 | prototype | 32 | True | False | 63.354 | 64.08 | 6 |
| unimts | 68.61 | prototype | 64 | True | False | 65.327 | 66.323 | 6 |
| unimts | 68.61 | prototype | 128 | True | False | 66.662 | 66.897 | 6 |
| unimts | 68.61 | ridge | 1 | True | False | 50.675 | 50.653 | 6 |
| unimts | 68.61 | ridge | 2 | True | False | 54.519 | 54.716 | 6 |
| unimts | 68.61 | ridge | 4 | True | False | 58.546 | 59.211 | 6 |
| unimts | 68.61 | ridge | 8 | True | False | 61.639 | 62.333 | 6 |
| unimts | 68.61 | ridge | 16 | True | False | 65.293 | 65.666 | 6 |
| unimts | 68.61 | ridge | 32 | True | False | 67.083 | 66.779 | 6 |
| unimts | 68.61 | ridge | 64 | True | False | 70.013 | 70.436 | 6 |
| unimts | 68.61 | ridge | 128 | True | False | 73.756 | 73.75 | 6 |

## 16-second windows

### inclusivehar
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | stream | n_devices | accuracy | f1_macro | balanced_accuracy | n_queries | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halo | 2.203 | halo-classifier | 0 | True | True | phone_waist | 1 | 40.94202898550724 | 36.78709733997359 | 40.36507814539749 | 552 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_waist | 1 | 38.224637681159415 | 34.740651032919196 | 37.67382773218873 | 552 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_waist | 1 | 35.688405797101446 | 34.99488572452391 | 35.64511722286581 | 552 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_waist | 1 | 35.688405797101446 | 34.99488572452391 | 35.64511722286581 | 552 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_waist | 1 | 35.32608695652174 | 34.67819119808324 | 35.215731492910265 | 552 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_waist | 1 | 39.85507246376812 | 37.95321961703227 | 39.57405597975332 | 552 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_waist | 1 | 35.14492753623188 | 34.159958152062345 | 35.11213131093183 | 552 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_waist | 1 | 36.05072463768116 | 35.90799191617225 | 36.10647881154881 | 552 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_waist | 1 | 37.68115942028986 | 36.85977143306598 | 37.74174830396116 | 552 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_waist | 1 | 37.5 | 37.1189783245824 | 37.659575322337 | 552 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_waist | 1 | 39.85507246376812 | 38.293327981393304 | 39.59497411716412 | 552 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_waist | 1 | 36.05072463768116 | 35.59763974767573 | 36.1474350645403 | 552 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_waist | 1 | 38.224637681159415 | 37.706695984340314 | 38.229979816590415 | 552 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_waist | 1 | 38.04347826086957 | 36.75825757921611 | 38.02631954368598 | 552 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_waist | 1 | 36.594202898550726 | 35.62561291695927 | 36.565280349669216 | 552 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_waist | 1 | 39.492753623188406 | 36.87775633129134 | 39.07612087744085 | 552 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_waist | 1 | 39.492753623188406 | 38.568901870106686 | 39.51574453164174 | 552 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_waist | 1 | 37.68115942028986 | 37.518657082732254 | 37.811488147525424 | 552 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_waist | 1 | 37.68115942028986 | 36.01281629302553 | 37.74158677916481 | 552 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_waist | 1 | 38.94927536231884 | 38.09187883608941 | 38.992374648571456 | 552 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_waist | 1 | 38.58695652173913 | 35.65383761977203 | 38.1391674295848 | 552 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_waist | 1 | 38.224637681159415 | 37.82865539941506 | 38.34091346636142 | 552 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_waist | 1 | 36.41304347826087 | 37.306231968695045 | 36.521982093912065 | 552 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_waist | 1 | 43.11594202898551 | 41.297545234312196 | 43.108883915054676 | 552 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_waist | 1 | 42.028985507246375 | 41.333669853094015 | 42.247458935185826 | 552 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_waist | 1 | 39.492753623188406 | 36.94806417035405 | 39.01412625108394 | 552 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_waist | 1 | 38.58695652173913 | 38.98512155529121 | 38.76322905167191 | 552 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_waist | 1 | 37.5 | 38.07277415292838 | 37.639947888356225 | 552 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_waist | 1 | 41.66666666666667 | 40.09574747777635 | 41.792942822898745 | 552 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_waist | 1 | 44.02173913043478 | 43.47753961373171 | 44.35581754227344 | 552 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_waist | 1 | 39.130434782608695 | 35.967702276360036 | 38.552958292831114 | 552 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_waist | 1 | 39.67391304347826 | 39.958928052721305 | 39.77730713718633 | 552 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_waist | 1 | 36.594202898550726 | 37.26683010159004 | 36.75554353243468 | 552 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_waist | 1 | 41.66666666666667 | 39.63890835182586 | 41.8599094800519 | 552 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_waist | 1 | 42.7536231884058 | 41.54065476565747 | 43.11864011241727 | 552 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_waist | 1 | 40.21739130434783 | 36.81697269444193 | 39.642032647400384 | 552 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_waist | 1 | 38.224637681159415 | 38.572874355519524 | 38.36367109285897 | 552 | ok |
| halo | 0.789 | all | 128 | True | True | phone_waist | 1 |  |  |  |  | n/a |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_waist | 1 | 31.88405797101449 | 27.605189005977543 | 31.62438262147526 | 552 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_waist | 1 | 28.6231884057971 | 28.69529847237256 | 28.75695222228933 | 552 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_waist | 1 | 28.6231884057971 | 28.69529847237256 | 28.75695222228933 | 552 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_waist | 1 | 30.79710144927536 | 30.889202738988637 | 30.878763944202888 | 552 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_waist | 1 | 32.608695652173914 | 32.27727345502784 | 32.719310096485756 | 552 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_waist | 1 | 29.891304347826086 | 29.751552910575697 | 29.963063286938624 | 552 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_waist | 1 | 31.15942028985507 | 30.616720987455075 | 31.29866441702277 | 552 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_waist | 1 | 33.15217391304348 | 33.579349394773516 | 33.25748430732145 | 552 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_waist | 1 | 31.15942028985507 | 31.228016866980084 | 31.187518983968737 | 552 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_waist | 1 | 31.70289855072464 | 31.52681743871077 | 31.854665512401503 | 552 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_waist | 1 | 29.528985507246375 | 29.664981343637898 | 29.663030301425348 | 552 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_waist | 1 | 27.173913043478258 | 27.515063237395683 | 27.32633984355857 | 552 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_waist | 1 | 30.978260869565215 | 30.700329570304802 | 31.240149087458224 | 552 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_waist | 1 | 27.173913043478258 | 27.48122743748769 | 27.356053429697507 | 552 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_waist | 1 | 31.34057971014493 | 31.48585079063901 | 31.246435527174647 | 552 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_waist | 1 | 33.69565217391305 | 33.51302848802214 | 33.89650259876241 | 552 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_waist | 1 | 27.173913043478258 | 27.486608238564518 | 27.215101012297044 | 552 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_waist | 1 | 28.804347826086957 | 29.168737041595787 | 28.902528634025305 | 552 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_waist | 1 | 33.33333333333333 | 32.16351044131287 | 33.802150619017986 | 552 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_waist | 1 | 24.818840579710145 | 25.172438272762438 | 24.962912867416943 | 552 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_waist | 1 | 29.347826086956523 | 28.62186191393607 | 29.519990021198833 | 552 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_waist | 1 | 38.04347826086957 | 36.12178219669082 | 38.63611060306853 | 552 | ok |
| harnet | 4.49 | all | 128 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_waist | 1 | 32.065217391304344 | 26.71670803520179 | 31.65556737175764 | 552 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_waist | 1 | 34.78260869565217 | 34.69980487902692 | 34.862708194357225 | 552 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_waist | 1 | 34.78260869565217 | 34.69980487902692 | 34.862708194357225 | 552 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_waist | 1 | 32.971014492753625 | 32.69785797148942 | 33.10562437456303 | 552 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_waist | 1 | 36.05072463768116 | 36.10066929980023 | 36.269729481488746 | 552 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_waist | 1 | 34.78260869565217 | 34.89042817632111 | 34.9350749336951 | 552 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_waist | 1 | 35.14492753623188 | 34.96014789698892 | 35.376596237299445 | 552 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_waist | 1 | 38.768115942028984 | 38.936682291409824 | 39.02170617054905 | 552 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_waist | 1 | 34.60144927536232 | 34.715945802374655 | 34.842730303535355 | 552 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_waist | 1 | 37.31884057971014 | 37.13481518100449 | 37.74331201206021 | 552 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_waist | 1 | 36.231884057971016 | 36.33474314742567 | 36.63353657124986 | 552 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_waist | 1 | 34.05797101449276 | 34.120592298575595 | 34.29533102645456 | 552 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_waist | 1 | 38.04347826086957 | 37.52433652707339 | 38.60622915643166 | 552 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_waist | 1 | 38.768115942028984 | 38.67883093356011 | 39.06738665009874 | 552 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_waist | 1 | 36.41304347826087 | 36.461895259507315 | 36.67292406181857 | 552 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_waist | 1 | 44.565217391304344 | 44.14022243759305 | 45.11421340755757 | 552 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_waist | 1 | 40.39855072463768 | 40.74718184174764 | 40.630533007339416 | 552 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_waist | 1 | 42.934782608695656 | 42.69292426240652 | 43.304894646964314 | 552 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_waist | 1 | 46.55797101449276 | 45.50874646211434 | 47.20714394061902 | 552 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_waist | 1 | 40.21739130434783 | 40.49657207514356 | 40.49570308447865 | 552 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_waist | 1 | 42.210144927536234 | 41.18087835207274 | 42.70178056279084 | 552 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_waist | 1 | 48.731884057971016 | 47.85737237681724 | 49.310155357951906 | 552 | ok |
| unimts | 68.61 | all | 128 | True | False | phone_waist | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_waist | 1 | 17.02898550724638 | 4.850361197110423 | 16.666666666666664 | 552 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_waist | 1 | 24.27536231884058 | 24.257395414459722 | 24.2412662069026 | 552 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_waist | 1 | 24.27536231884058 | 24.257395414459722 | 24.2412662069026 | 552 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_waist | 1 | 24.27536231884058 | 23.970517287816108 | 24.30830625853131 | 552 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_waist | 1 | 22.82608695652174 | 22.898511673451637 | 22.89571176083043 | 552 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_waist | 1 | 25.543478260869566 | 25.518116962905186 | 25.623377579879886 | 552 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_waist | 1 | 24.456521739130434 | 23.756021526398243 | 24.593330251761987 | 552 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_waist | 1 | 21.3768115942029 | 21.25800955065673 | 21.44470689435669 | 552 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_waist | 1 | 25.0 | 24.879396853124387 | 25.11464886246251 | 552 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_waist | 1 | 25.36231884057971 | 24.241186207604184 | 25.427947383322504 | 552 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_waist | 1 | 23.369565217391305 | 23.40382493795139 | 23.38734995884072 | 552 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_waist | 1 | 25.0 | 24.561157361359232 | 25.297682226665664 | 552 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_waist | 1 | 24.818840579710145 | 23.252492286656405 | 25.07131351793546 | 552 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_waist | 1 | 26.44927536231884 | 26.55247483384468 | 26.533574505729142 | 552 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_waist | 1 | 25.905797101449274 | 25.397199142376454 | 26.08513746351027 | 552 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_waist | 1 | 27.898550724637683 | 26.01563263858115 | 28.10346212063421 | 552 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_waist | 1 | 24.456521739130434 | 24.770149597431 | 24.56749689130616 | 552 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_waist | 1 | 21.920289855072465 | 20.597221289492996 | 22.203474003278103 | 552 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_waist | 1 | 23.55072463768116 | 20.460955862781027 | 23.91569998810027 | 552 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_waist | 1 | 25.724637681159418 | 25.882872136061795 | 25.756035994416273 | 552 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_waist | 1 | 24.637681159420293 | 22.47544595628379 | 25.201342616317856 | 552 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_waist | 1 | 23.18840579710145 | 17.38026598599065 | 23.84927891315923 | 552 | ok |
| normwear | 1293.86 | all | 128 | True | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_waist | 1 | 30.79710144927536 | 27.97260151286952 | 31.037923886554548 | 552 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_waist | 1 | 24.27536231884058 | 24.471786181009737 | 24.29227650641841 | 552 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_waist | 1 | 24.27536231884058 | 24.471786181009737 | 24.29227650641841 | 552 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_waist | 1 | 25.36231884057971 | 24.93801596317332 | 25.45550077997469 | 552 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_waist | 1 | 25.181159420289855 | 25.75744211780686 | 25.177272289403916 | 552 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_waist | 1 | 21.73913043478261 | 21.994421522551036 | 21.771582681318456 | 552 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_waist | 1 | 24.456521739130434 | 23.81210562619848 | 24.51332379532614 | 552 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_waist | 1 | 27.717391304347828 | 27.61967776351942 | 27.680921653511614 | 552 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_waist | 1 | 27.898550724637683 | 26.971160039626895 | 27.796543224135167 | 552 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_waist | 1 | 28.985507246376812 | 26.455575303403318 | 28.974970120404244 | 552 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_waist | 1 | 27.536231884057973 | 27.373995764177323 | 27.65645054008225 | 552 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_waist | 1 | 25.724637681159418 | 24.97052716308995 | 25.71180646171512 | 552 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_waist | 1 | 35.14492753623188 | 30.695536709803843 | 35.416238650872 | 552 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_waist | 1 | 26.08695652173913 | 26.4511707152229 | 26.13327939871601 | 552 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_waist | 1 | 26.44927536231884 | 24.373105946330266 | 26.626486857462687 | 552 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_waist | 1 | 33.51449275362319 | 28.16064754138149 | 33.90773892991339 | 552 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_waist | 1 | 30.978260869565215 | 31.267278366275313 | 31.082124588254146 | 552 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_waist | 1 | 29.347826086956523 | 26.299251523867145 | 29.574923187804973 | 552 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_waist | 1 | 34.78260869565217 | 26.798153397835645 | 35.31207964577705 | 552 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_waist | 1 | 29.347826086956523 | 29.703127406473957 | 29.465273407448798 | 552 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_waist | 1 | 27.355072463768117 | 22.776547322400155 | 27.723466886220795 | 552 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_waist | 1 | 36.231884057971016 | 27.34894875058523 | 36.857176016354025 | 552 | ok |
| limubert_x | 0.055 | all | 128 | False | False | phone_waist | 1 |  |  |  |  | n/a |

### motionsense
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | stream | n_devices | accuracy | f1_macro | balanced_accuracy | n_queries | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halo | 2.203 | halo-classifier | 0 | True | True | phone_front_pocket | 1 | 78.12177502579979 | 70.02128341899306 | 73.34725180117654 | 1938 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_front_pocket | 1 | 71.671826625387 | 63.20693478364213 | 68.77164602296298 | 1938 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_front_pocket | 1 | 76.41898864809082 | 72.85267255059502 | 73.80537162678023 | 1938 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_front_pocket | 1 | 76.41898864809082 | 72.85267255059502 | 73.80537162678023 | 1938 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_front_pocket | 1 | 77.70897832817337 | 73.98096883325648 | 75.14927724132986 | 1938 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_front_pocket | 1 | 81.16615067079464 | 76.250420742818 | 77.72565549549357 | 1938 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_front_pocket | 1 | 76.31578947368422 | 72.58141215564115 | 73.79239463463806 | 1938 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_front_pocket | 1 | 80.95975232198143 | 77.59605792005496 | 78.58095162160623 | 1938 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_front_pocket | 1 | 81.94014447884416 | 78.3726499382491 | 79.268887451768 | 1938 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_front_pocket | 1 | 83.02373581011352 | 79.36211547550447 | 80.31871183874073 | 1938 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_front_pocket | 1 | 83.95252837977296 | 79.71155322754488 | 80.8368667714394 | 1938 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_front_pocket | 1 | 81.42414860681114 | 77.9220869484741 | 78.88351388496005 | 1938 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_front_pocket | 1 | 83.74613003095975 | 80.83140658175732 | 81.71408351373414 | 1938 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_front_pocket | 1 | 85.39731682146542 | 82.46872158388248 | 83.25126111552689 | 1938 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_front_pocket | 1 | 87.25490196078431 | 84.39987162181359 | 85.08543521138598 | 1938 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_front_pocket | 1 | 85.39731682146542 | 81.35357173144168 | 82.1087653233966 | 1938 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_front_pocket | 1 | 84.6233230134159 | 81.53788719057955 | 82.50000847658319 | 1938 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_front_pocket | 1 | 86.37770897832817 | 84.25428835306349 | 85.30534721118427 | 1938 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_front_pocket | 1 | 87.51289989680082 | 84.90106916745256 | 85.59842752848986 | 1938 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_front_pocket | 1 | 89.78328173374614 | 87.51609331999106 | 87.98916613660403 | 1938 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_front_pocket | 1 | 86.9453044375645 | 83.69246554592547 | 84.21292622753408 | 1938 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_front_pocket | 1 | 88.13209494324046 | 85.84773429111277 | 86.58132353958621 | 1938 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_front_pocket | 1 | 89.2156862745098 | 86.96885582001605 | 87.60253218304125 | 1938 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_front_pocket | 1 | 89.2156862745098 | 86.63182716397841 | 87.02011313365897 | 1938 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_front_pocket | 1 | 91.22807017543859 | 89.1584463529805 | 89.45789125223159 | 1938 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_front_pocket | 1 | 88.44169246646027 | 85.1876223905308 | 85.49562408816888 | 1938 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_front_pocket | 1 | 90.04127966976264 | 87.77573274053104 | 88.20691381304496 | 1938 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_front_pocket | 1 | 91.48606811145511 | 89.74935069253172 | 90.26149151237767 | 1938 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_front_pocket | 1 | 90.71207430340557 | 88.68931472358366 | 89.2187695725133 | 1938 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_front_pocket | 1 | 92.51805985552114 | 90.80975338253836 | 91.11576707033167 | 1938 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_front_pocket | 1 | 89.2156862745098 | 86.11854856298774 | 86.31770225093965 | 1938 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_front_pocket | 1 | 92.51805985552114 | 90.60828435834534 | 90.72681096352005 | 1938 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_front_pocket | 1 | 94.16924664602682 | 92.54895803740575 | 92.75635922913125 | 1938 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_front_pocket | 1 | 90.71207430340557 | 88.75120712561436 | 89.15333981683766 | 1938 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_front_pocket | 1 | 93.13725490196079 | 91.70712518418502 | 92.04697302119101 | 1938 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_front_pocket | 1 | 89.62848297213623 | 86.57154388273428 | 86.81532213592446 | 1938 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_front_pocket | 1 | 93.65325077399382 | 91.85740785793014 | 91.7949939993678 | 1938 | ok |
| halo | 0.789 | 1nn | 128 | True | True | phone_front_pocket | 1 | 95.14963880288958 | 93.82301514820068 | 94.0848310368174 | 1938 | ok |
| halo | 0.789 | prototype | 128 | True | True | phone_front_pocket | 1 | 91.02167182662538 | 89.17950360676386 | 89.62682256677319 | 1938 | ok |
| halo | 0.789 | ridge | 128 | True | True | phone_front_pocket | 1 | 94.11764705882352 | 92.80047555284506 | 92.99368231064203 | 1938 | ok |
| halo | 2.203 | halo-classifier | 128 | True | True | phone_front_pocket | 1 | 90.09287925696594 | 87.3201763213544 | 87.40774478748592 | 1938 | ok |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | phone_front_pocket | 1 | 93.85964912280701 | 92.23112617387713 | 92.15238000150879 | 1938 | ok |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_front_pocket | 1 | 54.90196078431373 | 50.40909737016257 | 52.428621220974804 | 1938 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_front_pocket | 1 | 63.261093911248715 | 61.27053498117525 | 62.34725764162993 | 1938 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_front_pocket | 1 | 63.261093911248715 | 61.27053498117525 | 62.34725764162993 | 1938 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_front_pocket | 1 | 63.98348813209495 | 61.25003053708967 | 61.811330327409785 | 1938 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_front_pocket | 1 | 70.38183694530443 | 69.31592605366986 | 70.17081895365956 | 1938 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_front_pocket | 1 | 68.98864809081527 | 67.42479910241273 | 68.31856083387092 | 1938 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_front_pocket | 1 | 70.27863777089783 | 67.69641412678399 | 68.37884574460725 | 1938 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_front_pocket | 1 | 72.65221878224975 | 71.91676269536173 | 72.58430120416065 | 1938 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_front_pocket | 1 | 73.58101135190918 | 72.12352840371352 | 72.85161047566649 | 1938 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_front_pocket | 1 | 73.58101135190918 | 71.64711856505018 | 72.04872009422972 | 1938 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_front_pocket | 1 | 75.54179566563467 | 75.58187721570376 | 76.22666742154699 | 1938 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_front_pocket | 1 | 75.49019607843137 | 75.18024346537163 | 75.93492700621253 | 1938 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_front_pocket | 1 | 78.2765737874097 | 77.49043584058808 | 77.62809118229522 | 1938 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_front_pocket | 1 | 78.84416924664602 | 79.18169633723818 | 79.8882948323437 | 1938 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_front_pocket | 1 | 78.53457172342621 | 78.23562573326566 | 78.94135173478766 | 1938 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_front_pocket | 1 | 81.88854489164086 | 81.93738430608633 | 82.159034216241 | 1938 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_front_pocket | 1 | 83.33333333333334 | 83.86483845282568 | 84.55236801739917 | 1938 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_front_pocket | 1 | 78.68937048503611 | 78.40164705613365 | 79.10601249961698 | 1938 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_front_pocket | 1 | 84.21052631578947 | 84.8698128121416 | 84.99178850827926 | 1938 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_front_pocket | 1 | 84.10732714138287 | 84.65086650532973 | 85.02245362451139 | 1938 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_front_pocket | 1 | 79.61816305469557 | 79.50632693847548 | 80.16655385153881 | 1938 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_front_pocket | 1 | 86.01651186790505 | 86.68408151854138 | 87.01643647496982 | 1938 | ok |
| harnet | 4.49 | 1nn | 128 | False | False | phone_front_pocket | 1 | 85.70691434468525 | 86.30019471234509 | 86.6802421814915 | 1938 | ok |
| harnet | 4.49 | prototype | 128 | False | False | phone_front_pocket | 1 | 79.72136222910217 | 79.60391557714362 | 80.25369590537953 | 1938 | ok |
| harnet | 4.49 | ridge | 128 | False | False | phone_front_pocket | 1 | 87.25490196078431 | 87.85923369806464 | 88.24105986395243 | 1938 | ok |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_front_pocket | 1 | 55.83075335397317 | 43.32274805560321 | 50.621711707112695 | 1938 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_front_pocket | 1 | 68.52425180598554 | 68.42524111492125 | 68.84809389877874 | 1938 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_front_pocket | 1 | 68.52425180598554 | 68.42524111492125 | 68.84809389877874 | 1938 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_front_pocket | 1 | 66.51186790505676 | 67.05199683908477 | 67.7363006101597 | 1938 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_front_pocket | 1 | 77.50257997936016 | 78.47584405651506 | 79.0196170858016 | 1938 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_front_pocket | 1 | 72.29102167182663 | 73.85940502898308 | 74.48237742008229 | 1938 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_front_pocket | 1 | 71.62022703818369 | 73.54970863886973 | 74.01985154945501 | 1938 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_front_pocket | 1 | 80.4437564499484 | 81.09461478947148 | 81.39882280479549 | 1938 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_front_pocket | 1 | 75.18059855521156 | 76.93437826044126 | 77.19716298362744 | 1938 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_front_pocket | 1 | 76.0061919504644 | 77.95623962939739 | 78.17175767263168 | 1938 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_front_pocket | 1 | 83.79772961816305 | 84.08368962485966 | 84.56147283315708 | 1938 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_front_pocket | 1 | 78.99896800825593 | 80.68667314690222 | 81.29649967045077 | 1938 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_front_pocket | 1 | 80.4437564499484 | 82.01489321337698 | 82.42002199496254 | 1938 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_front_pocket | 1 | 86.17131062951496 | 86.45222801476956 | 86.75985389575455 | 1938 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_front_pocket | 1 | 81.26934984520123 | 82.7461374890107 | 83.38494919203305 | 1938 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_front_pocket | 1 | 82.81733746130031 | 84.45096401693255 | 85.04608220427609 | 1938 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_front_pocket | 1 | 87.25490196078431 | 87.38522266455226 | 87.7064514445698 | 1938 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_front_pocket | 1 | 81.21775025799793 | 82.69947234921673 | 83.36863525774186 | 1938 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_front_pocket | 1 | 83.90092879256966 | 85.37297288694754 | 85.77316119443688 | 1938 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_front_pocket | 1 | 90.45407636738906 | 90.34371078490621 | 90.69729774467783 | 1938 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_front_pocket | 1 | 81.73374613003097 | 83.10688055800485 | 83.77943715012096 | 1938 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_front_pocket | 1 | 85.55211558307533 | 86.94679383674112 | 87.24793291864246 | 1938 | ok |
| unimts | 68.61 | 1nn | 128 | True | False | phone_front_pocket | 1 | 90.35087719298247 | 90.59227195524954 | 90.68603402979728 | 1938 | ok |
| unimts | 68.61 | prototype | 128 | True | False | phone_front_pocket | 1 | 81.88854489164086 | 83.44295317127943 | 84.22587449311003 | 1938 | ok |
| unimts | 68.61 | ridge | 128 | True | False | phone_front_pocket | 1 | 87.56449948400413 | 88.68438574290552 | 89.01662571886449 | 1938 | ok |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_front_pocket | 1 | 22.445820433436534 | 6.110408765276022 | 16.666666666666664 | 1938 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_front_pocket | 1 | 25.644994840041278 | 24.81432669772131 | 25.46188360060046 | 1938 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_front_pocket | 1 | 25.644994840041278 | 24.81432669772131 | 25.46188360060046 | 1938 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_front_pocket | 1 | 25.696594427244584 | 23.524593663155777 | 24.701252972191178 | 1938 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_front_pocket | 1 | 27.708978328173373 | 26.674154393896725 | 27.12967522644548 | 1938 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_front_pocket | 1 | 28.53457172342621 | 27.05926833727795 | 27.543941212446327 | 1938 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_front_pocket | 1 | 26.367389060887515 | 23.233771888023792 | 25.231536202448655 | 1938 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_front_pocket | 1 | 31.475748194014447 | 30.853003101431526 | 31.765234559043492 | 1938 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_front_pocket | 1 | 28.431372549019606 | 26.611842463970557 | 27.70382311674699 | 1938 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_front_pocket | 1 | 25.79979360165119 | 21.52118486025549 | 25.095532323415952 | 1938 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_front_pocket | 1 | 33.33333333333333 | 32.609792450149854 | 33.287300597267816 | 1938 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_front_pocket | 1 | 31.166150670794636 | 28.921788316042154 | 30.66930865325271 | 1938 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_front_pocket | 1 | 27.347781217750256 | 21.817787021508042 | 26.98842354789886 | 1938 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_front_pocket | 1 | 38.13209494324045 | 37.79881244668961 | 38.52485879486616 | 1938 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_front_pocket | 1 | 30.392156862745097 | 27.636391191314395 | 29.590460577026494 | 1938 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_front_pocket | 1 | 29.51496388028896 | 23.464200568555217 | 30.16089810551052 | 1938 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_front_pocket | 1 | 43.4468524251806 | 43.60420833175153 | 44.65605813985319 | 1938 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_front_pocket | 1 | 33.178534571723425 | 30.759543178550274 | 32.859998706190396 | 1938 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_front_pocket | 1 | 31.269349845201237 | 25.26186932856971 | 33.13360919363714 | 1938 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_front_pocket | 1 | 46.749226006191954 | 47.17000889797511 | 48.38637144551829 | 1938 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_front_pocket | 1 | 31.11455108359133 | 28.162170039840444 | 30.81908314204677 | 1938 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_front_pocket | 1 | 32.765737874097006 | 27.44934847654677 | 35.31353991778454 | 1938 | ok |
| normwear | 1293.86 | 1nn | 128 | True | False | phone_front_pocket | 1 | 53.92156862745098 | 54.34531197170188 | 55.828119614160556 | 1938 | ok |
| normwear | 1293.86 | prototype | 128 | True | False | phone_front_pocket | 1 | 31.630546955624357 | 28.735082054301625 | 31.73684459510952 | 1938 | ok |
| normwear | 1293.86 | ridge | 128 | True | False | phone_front_pocket | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_front_pocket | 1 | 49.43240454076368 | 42.68406772213868 | 49.06070748142958 | 1938 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_front_pocket | 1 | 57.89473684210527 | 56.00824139368418 | 57.04195898306846 | 1938 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_front_pocket | 1 | 57.89473684210527 | 56.00824139368418 | 57.04195898306846 | 1938 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_front_pocket | 1 | 53.61197110423117 | 50.16362351982067 | 54.33527647326413 | 1938 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_front_pocket | 1 | 67.3890608875129 | 65.09355730720856 | 65.57185296878174 | 1938 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_front_pocket | 1 | 64.49948400412796 | 62.440745450943616 | 63.22619461451483 | 1938 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_front_pocket | 1 | 56.604747162022704 | 51.91596462338281 | 57.29299407750199 | 1938 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_front_pocket | 1 | 75.85139318885449 | 73.65009414011794 | 74.21301797472269 | 1938 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_front_pocket | 1 | 67.69865841073272 | 65.50190259629501 | 66.43807478582646 | 1938 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_front_pocket | 1 | 60.06191950464397 | 56.65991600053412 | 62.28528454276111 | 1938 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_front_pocket | 1 | 80.4437564499484 | 78.39649655376127 | 78.7422227598404 | 1938 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_front_pocket | 1 | 70.53663570691434 | 68.95702345453432 | 70.1003043520716 | 1938 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_front_pocket | 1 | 64.39628482972137 | 61.266383165293995 | 67.42695579329535 | 1938 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_front_pocket | 1 | 85.29411764705883 | 83.26429301268257 | 83.76504281799218 | 1938 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_front_pocket | 1 | 74.87100103199175 | 73.82384538268623 | 74.64132221251649 | 1938 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_front_pocket | 1 | 69.45304437564499 | 66.39424810241677 | 73.42706374732174 | 1938 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_front_pocket | 1 | 89.1124871001032 | 87.59510893516261 | 87.56690580422436 | 1938 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_front_pocket | 1 | 75.69659442724458 | 74.97419752881063 | 75.99665065646383 | 1938 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_front_pocket | 1 | 73.42621259029927 | 69.95945773641944 | 77.83718698954148 | 1938 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_front_pocket | 1 | 91.3312693498452 | 90.03308190742497 | 90.16015766516664 | 1938 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_front_pocket | 1 | 76.67698658410733 | 75.88236050323215 | 76.70566891334823 | 1938 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_front_pocket | 1 | 76.52218782249743 | 73.20107206835225 | 81.11377824007985 | 1938 | ok |
| limubert_x | 0.055 | 1nn | 128 | False | False | phone_front_pocket | 1 | 92.72445820433437 | 91.6719466655783 | 91.72378278030052 | 1938 | ok |
| limubert_x | 0.055 | prototype | 128 | False | False | phone_front_pocket | 1 | 76.88338493292053 | 76.15893102551154 | 76.78325819310837 | 1938 | ok |
| limubert_x | 0.055 | ridge | 128 | False | False | phone_front_pocket | 1 | 77.19298245614034 | 74.02940795691455 | 81.85246986736696 | 1938 | ok |

### realworld
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | stream | n_devices | accuracy | f1_macro | balanced_accuracy | n_queries | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halo | 2.203 | halo-classifier | 0 | True | True | phone_waist | 1 | 46.658932714617166 | 40.1495758263019 | 52.32330937839447 | 4310 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_waist | 1 | 30.185614849187935 | 21.367369286428843 | 35.816322858788105 | 4310 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_waist | 1 | 61.39211136890951 | 61.13946640742702 | 63.97509040950131 | 4310 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_waist | 1 | 61.39211136890951 | 61.13946640742702 | 63.97509040950131 | 4310 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_waist | 1 | 61.7169373549884 | 60.591873746178074 | 64.4129568778625 | 4310 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_waist | 1 | 62.01856148491879 | 60.44350052228015 | 65.35934631907384 | 4310 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_waist | 1 | 62.5522041763341 | 62.13249971677466 | 64.60034944522981 | 4310 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_waist | 1 | 69.37354988399072 | 69.52542855482197 | 71.4314864361323 | 4310 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_waist | 1 | 68.02784222737819 | 67.78359132293424 | 70.5358720905554 | 4310 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_waist | 1 | 69.67517401392112 | 68.5153244601563 | 72.09594181937848 | 4310 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_waist | 1 | 68.88631090487239 | 67.58689818630077 | 71.75176720897373 | 4310 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_waist | 1 | 70.97447795823666 | 71.40015907900681 | 72.76764533680435 | 4310 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_waist | 1 | 72.50580046403712 | 72.76873582120477 | 74.6124280972739 | 4310 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_waist | 1 | 72.38979118329466 | 72.67458413113054 | 74.53610391352198 | 4310 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_waist | 1 | 73.89791183294663 | 73.44123837021318 | 76.03097995787547 | 4310 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_waist | 1 | 70.88167053364269 | 69.50367386213885 | 73.55111715661003 | 4310 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_waist | 1 | 74.24593967517401 | 74.66361189380393 | 75.89061316336408 | 4310 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_waist | 1 | 74.75638051044083 | 75.22548414663213 | 76.8733987965949 | 4310 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_waist | 1 | 76.98375870069606 | 77.5723858759559 | 78.58791122504847 | 4310 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_waist | 1 | 78.72389791183295 | 78.4544087932683 | 80.34955109886207 | 4310 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_waist | 1 | 74.50116009280742 | 73.00423785797123 | 76.98235275722772 | 4310 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_waist | 1 | 77.65661252900232 | 78.25534034552541 | 79.20942199051449 | 4310 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_waist | 1 | 76.61252900232019 | 77.4764905494785 | 78.4875572586391 | 4310 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_waist | 1 | 80.0 | 80.7545404459088 | 81.3160335298158 | 4310 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_waist | 1 | 80.78886310904872 | 80.81767587812641 | 82.33754121275372 | 4310 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_waist | 1 | 75.31322505800463 | 73.96823648215529 | 77.53538521038874 | 4310 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_waist | 1 | 79.67517401392111 | 80.49201126020236 | 81.21642581588968 | 4310 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_waist | 1 | 77.7262180974478 | 78.39111377403752 | 79.61608983097031 | 4310 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_waist | 1 | 81.1368909512761 | 82.13157914367868 | 82.51606745727975 | 4310 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_waist | 1 | 82.06496519721578 | 82.25691372594292 | 83.48091368564675 | 4310 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_waist | 1 | 75.80046403712298 | 74.33042069273078 | 78.30635942462753 | 4310 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_waist | 1 | 80.48723897911833 | 81.30623116586753 | 82.00940894823657 | 4310 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_waist | 1 | 78.49187935034803 | 79.41918149219454 | 80.11277182967656 | 4310 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_waist | 1 | 82.04176334106728 | 82.93835450540674 | 83.31034194415527 | 4310 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_waist | 1 | 83.10904872389791 | 83.33459049530028 | 84.5459621513843 | 4310 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_waist | 1 | 76.33410672853829 | 74.9577825883594 | 78.69319974908005 | 4310 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_waist | 1 | 81.53132250580046 | 82.44301826973616 | 82.86391232809879 | 4310 | ok |
| halo | 0.789 | all | 128 | True | True | phone_waist | 1 |  |  |  |  | n/a |
| halo | 2.203 | halo-classifier | 0 | True | True | phone_forearm | 1 | 52.20030349013657 | 51.6929085921919 | 57.08162492773363 | 1977 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_forearm | 1 | 40.26302478502782 | 28.321276570380192 | 36.530795296412215 | 1977 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_forearm | 1 | 48.5078401618614 | 48.52243412886894 | 50.76312430235795 | 1977 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_forearm | 1 | 48.5078401618614 | 48.52243412886894 | 50.76312430235795 | 1977 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_forearm | 1 | 49.6206373292868 | 50.4167419200391 | 53.263863178880555 | 1977 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_forearm | 1 | 52.50379362670713 | 53.89089361929997 | 57.53625952204533 | 1977 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_forearm | 1 | 48.96307536671725 | 49.36511191027715 | 52.016347255006124 | 1977 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_forearm | 1 | 54.375316135558926 | 55.43584896965794 | 56.422081356989786 | 1977 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_forearm | 1 | 53.667172483560954 | 55.44145849373117 | 56.91342282856273 | 1977 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_forearm | 1 | 55.488113302984324 | 56.24680858182214 | 58.35217840276334 | 1977 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_forearm | 1 | 56.145675265553876 | 58.89538482998511 | 61.42800041777075 | 1977 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_forearm | 1 | 54.77996965098634 | 56.18083497508841 | 57.3291211097924 | 1977 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_forearm | 1 | 56.297420333839156 | 58.30943617600759 | 59.429150363877525 | 1977 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_forearm | 1 | 59.5852301466869 | 61.163848738822345 | 62.41821241163845 | 1977 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_forearm | 1 | 61.153262518968134 | 61.816778793202346 | 64.17893679870886 | 1977 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_forearm | 1 | 59.180576631259484 | 62.51704904726333 | 64.13548769635715 | 1977 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_forearm | 1 | 58.573596358118365 | 60.224318287205946 | 61.90135136016088 | 1977 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_forearm | 1 | 59.28174001011634 | 61.661982141952606 | 63.49792830626334 | 1977 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_forearm | 1 | 62.36722306525038 | 63.84581101189093 | 65.80456465720621 | 1977 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_forearm | 1 | 62.569549822964085 | 62.43826730959029 | 65.45155771703539 | 1977 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_forearm | 1 | 61.05209914011128 | 63.96766430851187 | 66.2110118136854 | 1977 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_forearm | 1 | 62.2154779969651 | 64.39104921538888 | 66.41770187737842 | 1977 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_forearm | 1 | 60.74860900354072 | 63.04892987764428 | 64.18322506222616 | 1977 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_forearm | 1 | 63.98583712696004 | 65.41566681877661 | 68.20680696465905 | 1977 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_forearm | 1 | 66.56550328780982 | 67.31684313429889 | 68.96117009597936 | 1977 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_forearm | 1 | 62.06373292867981 | 65.38624967838771 | 67.35323517340261 | 1977 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_forearm | 1 | 64.59281740010117 | 66.50450348011819 | 68.72249484423033 | 1977 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_forearm | 1 | 60.771208226221084 | 51.272658210746016 | 60.07260596977332 | 1945 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_forearm | 1 | 64.42159383033419 | 61.758657738124235 | 65.16205017866224 | 1945 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_forearm | 1 | 68.48329048843188 | 58.012261830850264 | 67.95470447804688 | 1945 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_forearm | 1 | 61.59383033419024 | 60.48801391039945 | 62.64764702890798 | 1945 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_forearm | 1 | 66.63239074550128 | 56.515207461943795 | 67.46096121586191 | 1945 | ok |
| halo | 0.789 | all | 64 | True | True | phone_forearm | 1 |  |  |  |  | n/a |
| halo | 0.789 | all | 128 | True | True | phone_forearm | 1 |  |  |  |  | n/a |
| halo | 2.203 | halo-classifier | 0 | True | True | phone_thigh | 1 | 42.57309941520468 | 40.93848566369049 | 43.177156680357704 | 1710 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_thigh | 1 | 21.637426900584796 | 15.46792239257433 | 25.06413941243285 | 1710 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_thigh | 1 | 52.10526315789473 | 50.51177160793847 | 54.117382633476964 | 1710 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_thigh | 1 | 52.10526315789473 | 50.51177160793847 | 54.117382633476964 | 1710 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_thigh | 1 | 52.33918128654971 | 50.443171550375766 | 54.53244952726193 | 1710 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_thigh | 1 | 49.53216374269006 | 48.3191049872454 | 51.98356865178604 | 1710 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_thigh | 1 | 52.28070175438596 | 50.49526705467346 | 54.532956490568885 | 1710 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_thigh | 1 | 59.29824561403508 | 57.81023844015334 | 61.53392995806701 | 1710 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_thigh | 1 | 56.14035087719298 | 54.12304433919082 | 59.04893267856366 | 1710 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_thigh | 1 | 58.36257309941521 | 56.46993433901552 | 61.54996150387151 | 1710 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_thigh | 1 | 57.19298245614035 | 56.28712852217941 | 60.16470170799448 | 1710 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_thigh | 1 | 59.94152046783626 | 58.77913053532489 | 62.18771923171565 | 1710 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_thigh | 1 | 63.50877192982456 | 62.67628915079053 | 65.71933554543897 | 1710 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_thigh | 1 | 62.92397660818714 | 61.67454013140799 | 66.48819623733814 | 1710 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_thigh | 1 | 66.90058479532163 | 66.15955244421002 | 69.79437530320796 | 1710 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_thigh | 1 | 60.76023391812866 | 60.319566071817185 | 64.00427466364147 | 1710 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_thigh | 1 | 66.49122807017544 | 64.93210342988827 | 68.28886610644476 | 1710 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_thigh | 1 | 65.2046783625731 | 65.04545825429008 | 68.05800690533373 | 1710 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_thigh | 1 | 66.25730994152048 | 64.90601319594506 | 68.82746289369179 | 1710 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_thigh | 1 | 71.69590643274853 | 70.98686525148706 | 73.60203159487804 | 1710 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_thigh | 1 | 64.6783625730994 | 64.1551818331161 | 67.34983700364259 | 1710 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_thigh | 1 | 70.05847953216374 | 69.39126029465481 | 71.93180882455981 | 1710 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_thigh | 1 | 67.71929824561404 | 68.4605382045734 | 70.60611910683627 | 1710 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_thigh | 1 | 69.23976608187135 | 69.19382320560366 | 71.2621020379274 | 1710 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_thigh | 1 | 74.85380116959064 | 75.8186328441961 | 76.98773333650001 | 1710 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_thigh | 1 | 67.19298245614034 | 67.86869810319588 | 69.42847916686937 | 1710 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_thigh | 1 | 72.98245614035088 | 73.18157237263392 | 74.4594876806421 | 1710 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_thigh | 1 | 66.3157894736842 | 66.74975941387578 | 69.64992713988065 | 1710 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_thigh | 1 | 70.93567251461988 | 72.15843462810226 | 73.00957350832233 | 1710 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_thigh | 1 | 76.31578947368422 | 77.35685744595493 | 78.50176464433815 | 1710 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_thigh | 1 | 66.4327485380117 | 66.98234370990296 | 69.46639147176728 | 1710 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_thigh | 1 | 72.63157894736842 | 72.56057772179616 | 74.5158243026614 | 1710 | ok |
| halo | 0.789 | all | 64 | True | True | phone_thigh | 1 |  |  |  |  | n/a |
| halo | 0.789 | all | 128 | True | True | phone_thigh | 1 |  |  |  |  | n/a |
| halo | 2.203 | halo-classifier | 0 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 58.604206500956025 | 64.64866787370568 | 67.37216756002282 | 1046 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| halo | 0.789 | 1nn | 1 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 65.10516252390057 | 62.488999256019454 | 65.91292027641506 | 1046 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 65.10516252390057 | 62.488999256019454 | 65.91292027641506 | 1046 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 67.11281070745699 | 65.18606403785145 | 69.45078986075175 | 1046 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 60.22944550669216 | 61.954663727667736 | 67.6343448289999 | 1046 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 65.86998087954112 | 63.771032839470855 | 67.75757547755308 | 1046 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 74.66539196940727 | 72.73197290341075 | 75.3520368734067 | 1046 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 68.64244741873804 | 65.0023089822452 | 69.69289515787193 | 1046 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 72.27533460803059 | 69.52048841086551 | 73.05662044877495 | 1046 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 66.1567877629063 | 68.59513779000747 | 72.44481302782185 | 1046 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 75.81261950286807 | 73.58553890075467 | 76.42466903698346 | 1046 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 74.37858508604207 | 73.4120221861151 | 74.98746164022334 | 1046 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 74.47418738049714 | 71.54084558044414 | 75.32680560055584 | 1046 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 76.76864244741873 | 75.15971043047942 | 77.89073898555397 | 1046 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 66.53919694072657 | 70.81395223833988 | 73.14523976798606 | 1046 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 76.8642447418738 | 75.63626621943403 | 77.50705341570105 | 1046 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 76.38623326959846 | 75.16859583045819 | 76.64958728242564 | 1046 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 76.38623326959846 | 75.1201908120028 | 77.22159914163376 | 1046 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 79.25430210325047 | 78.02684358939528 | 79.67347261097095 | 1046 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 69.69407265774379 | 73.55785792329172 | 74.68288984775955 | 1046 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 79.34990439770554 | 78.43911381020533 | 80.0088566771772 | 1046 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 75.14619883040936 | 71.087240723453 | 72.19010825107375 | 1026 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 76.21832358674465 | 71.14717994885706 | 73.6164209116734 | 1026 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 80.4093567251462 | 66.91123886798229 | 77.40232174243198 | 1026 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 70.37037037037037 | 62.604408561512685 | 72.50379639533982 | 1026 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_forearm+phone_thigh+phone_waist | 3 | 79.23976608187134 | 75.72667014445324 | 77.2274231233591 | 1026 | ok |
| halo | 0.789 | all | 32 | True | True | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| halo | 0.789 | all | 64 | True | True | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| halo | 0.789 | all | 128 | True | True | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_waist | 1 | 36.17169373549884 | 29.182664968288062 | 38.80874011231911 | 4310 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_waist | 1 | 45.243619489559165 | 42.63851766946554 | 43.54634506301001 | 4310 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_waist | 1 | 45.243619489559165 | 42.63851766946554 | 43.54634506301001 | 4310 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_waist | 1 | 44.733178654292345 | 41.90118001433015 | 43.20739907521502 | 4310 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_waist | 1 | 50.556844547563806 | 48.05633810912762 | 49.183576527971454 | 4310 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_waist | 1 | 51.11368909512761 | 48.841055495240816 | 50.38040182959889 | 4310 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_waist | 1 | 51.36890951276102 | 48.606086876740825 | 49.967516329925694 | 4310 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_waist | 1 | 55.54524361948956 | 53.838091156041145 | 55.16819461928454 | 4310 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_waist | 1 | 57.21577726218098 | 55.38089372246971 | 57.37913019858149 | 4310 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_waist | 1 | 57.911832946635734 | 55.344327648179146 | 57.232049241382185 | 4310 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_waist | 1 | 57.4245939675174 | 55.64657548998333 | 56.88155879043415 | 4310 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_waist | 1 | 60.48723897911833 | 58.73784366323418 | 60.47414119221022 | 4310 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_waist | 1 | 62.691415313225065 | 60.92005822052515 | 62.67071607782094 | 4310 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_waist | 1 | 61.229698375870065 | 60.32279085471439 | 61.55732330386603 | 4310 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_waist | 1 | 63.82830626450116 | 63.00648059637943 | 64.1197224461855 | 4310 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_waist | 1 | 67.63341067285383 | 66.96373080407531 | 68.0649491828988 | 4310 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_waist | 1 | 64.24593967517401 | 63.755713824622504 | 64.66089223217223 | 4310 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_waist | 1 | 65.63805104408353 | 64.67921655037884 | 65.6382453367076 | 4310 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_waist | 1 | 70.62645011600928 | 71.10030814658566 | 71.45705183386717 | 4310 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_waist | 1 | 65.38283062645012 | 64.81401198578381 | 65.50974314261343 | 4310 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_waist | 1 | 66.47331786542924 | 65.7990820908256 | 66.58869491262998 | 4310 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_waist | 1 | 73.20185614849188 | 73.79945782435551 | 74.09169132327077 | 4310 | ok |
| harnet | 4.49 | all | 128 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_forearm | 1 | 40.414769853313096 | 29.30002132428943 | 38.12087293366914 | 1977 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_forearm | 1 | 41.274658573596355 | 38.03072420170407 | 40.720309508303004 | 1977 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_forearm | 1 | 41.274658573596355 | 38.03072420170407 | 40.720309508303004 | 1977 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_forearm | 1 | 42.943854324734446 | 39.41795437943855 | 42.73237436753046 | 1977 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_forearm | 1 | 47.040971168437025 | 43.76460086368067 | 47.18227333833403 | 1977 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_forearm | 1 | 45.52352048558422 | 41.69901709555944 | 44.77139039818816 | 1977 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_forearm | 1 | 46.78806272129489 | 42.66809235982273 | 46.896162105380526 | 1977 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_forearm | 1 | 48.91249367728882 | 46.45295493071786 | 49.52451807570467 | 1977 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_forearm | 1 | 50.42994436014163 | 47.17599787251318 | 50.22393163800655 | 1977 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_forearm | 1 | 51.18866970156803 | 47.67978667948309 | 51.309757708102 | 1977 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_forearm | 1 | 49.36772888214466 | 47.10242626019048 | 51.08302576389394 | 1977 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_forearm | 1 | 53.414264036418814 | 50.108640679862674 | 53.2656155101066 | 1977 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_forearm | 1 | 55.74102175012645 | 52.457268510676855 | 56.086307469991766 | 1977 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_forearm | 1 | 53.46484572584724 | 52.70064345207296 | 55.718471959399096 | 1977 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_forearm | 1 | 54.17298937784521 | 51.53583475378394 | 55.03420787359388 | 1977 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_forearm | 1 | 57.91603439554881 | 56.18656432013438 | 59.54459632923531 | 1977 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_forearm | 1 | 56.45244215938303 | 49.018318168211046 | 57.23609539967346 | 1945 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_forearm | 1 | 56.709511568123396 | 47.516137888525336 | 56.67850477542598 | 1945 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_forearm | 1 | 60.66838046272493 | 51.62734703353017 | 60.97733558087883 | 1945 | ok |
| harnet | 4.49 | all | 64 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| harnet | 4.49 | all | 128 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_thigh | 1 | 36.0233918128655 | 29.46245418050398 | 42.751127210818666 | 1710 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_thigh | 1 | 43.918128654970765 | 43.5079562901161 | 46.85237674411462 | 1710 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_thigh | 1 | 43.918128654970765 | 43.5079562901161 | 46.85237674411462 | 1710 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_thigh | 1 | 44.73684210526316 | 44.929706200483885 | 48.62724864686795 | 1710 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_thigh | 1 | 46.49122807017544 | 46.696161296876994 | 50.53958097423321 | 1710 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_thigh | 1 | 48.36257309941521 | 48.331191893387135 | 51.934342030813596 | 1710 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_thigh | 1 | 49.590643274853804 | 50.22026259067519 | 54.61423635275977 | 1710 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_thigh | 1 | 50.29239766081871 | 51.529909442177654 | 55.716372402869354 | 1710 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_thigh | 1 | 52.046783625730995 | 52.926686925450575 | 57.28015521838461 | 1710 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_thigh | 1 | 54.853801169590646 | 56.497708231056464 | 61.00286925962215 | 1710 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_thigh | 1 | 52.69005847953216 | 53.44839758462313 | 57.02627749529522 | 1710 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_thigh | 1 | 54.327485380116954 | 55.85385373104591 | 61.23964722076873 | 1710 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_thigh | 1 | 57.134502923976605 | 59.67794559413757 | 64.1301595329916 | 1710 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_thigh | 1 | 54.327485380116954 | 55.24951236113915 | 58.28140570447623 | 1710 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_thigh | 1 | 55.73099415204679 | 57.46172887586234 | 62.23676356964596 | 1710 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_thigh | 1 | 59.29824561403508 | 61.93803749324367 | 65.55533136259585 | 1710 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_thigh | 1 | 55.43859649122807 | 56.09607825581347 | 59.07659749461399 | 1710 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_thigh | 1 | 57.42690058479533 | 59.5261230811917 | 63.77206726724985 | 1710 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_thigh | 1 | 60.23391812865497 | 63.00604719479621 | 65.92557944514645 | 1710 | ok |
| harnet | 4.49 | all | 64 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| harnet | 4.49 | all | 128 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| harnet | 4.49 | zero_support | 0 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| harnet | 4.49 | 1nn | 1 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 52.29445506692161 | 51.36437282804201 | 56.21510254169132 | 1046 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 52.29445506692161 | 51.36437282804201 | 56.21510254169132 | 1046 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 52.00764818355641 | 50.359034752830965 | 54.73529187388044 | 1046 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 58.69980879541109 | 56.759289634261755 | 59.41612201307007 | 1046 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 53.82409177820268 | 50.901303923838405 | 55.447576780476204 | 1046 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 52.868068833652 | 50.33646352198057 | 54.97920742905744 | 1046 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 62.81070745697897 | 63.409793618336394 | 65.23792640417243 | 1046 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 57.45697896749522 | 55.966569220504425 | 62.19422100016318 | 1046 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 58.604206500956025 | 58.10027287279378 | 63.14136257578193 | 1046 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 65.10516252390057 | 65.02159034628733 | 67.21038201010802 | 1046 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 61.66347992351816 | 60.48262206982902 | 66.45426693137051 | 1046 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 63.38432122370937 | 64.05294224586474 | 69.46931034571871 | 1046 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 66.76413255360623 | 58.76239980791269 | 68.58998367596666 | 1026 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 62.28070175438597 | 54.24364605152425 | 64.65835097468387 | 1026 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_forearm+phone_thigh+phone_waist | 3 | 65.59454191033139 | 57.14103592935971 | 68.08465292297005 | 1026 | ok |
| harnet | 4.49 | all | 32 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| harnet | 4.49 | all | 64 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| harnet | 4.49 | all | 128 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_waist | 1 | 42.343387470997676 | 33.089099411635324 | 44.288051945805485 | 4310 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_waist | 1 | 59.62877030162414 | 59.56183712217877 | 61.85427154282591 | 4310 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_waist | 1 | 59.62877030162414 | 59.56183712217877 | 61.85427154282591 | 4310 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_waist | 1 | 58.79350348027842 | 57.86427005253253 | 61.03275498562341 | 4310 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_waist | 1 | 68.0046403712297 | 68.66041640313509 | 70.09398744133772 | 4310 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_waist | 1 | 64.29234338747099 | 64.76951173078243 | 66.90932358835072 | 4310 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_waist | 1 | 63.54988399071926 | 63.16259684621026 | 66.01302365372955 | 4310 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_waist | 1 | 70.25522041763341 | 71.14409558007598 | 72.02535611055904 | 4310 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_waist | 1 | 67.70301624129931 | 69.08876590253803 | 70.36744041593937 | 4310 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_waist | 1 | 67.4245939675174 | 68.50899486789639 | 70.01334397702261 | 4310 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_waist | 1 | 73.50348027842227 | 74.62077278678034 | 75.73551402707504 | 4310 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_waist | 1 | 70.20881670533643 | 72.20808316645059 | 72.4034233653256 | 4310 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_waist | 1 | 72.0417633410673 | 72.53937559800082 | 74.26930912563674 | 4310 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_waist | 1 | 74.89559164733178 | 75.77927016045061 | 76.61993423247434 | 4310 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_waist | 1 | 73.54988399071925 | 75.106411696015 | 75.42084547025071 | 4310 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_waist | 1 | 74.75638051044083 | 74.26691268685721 | 76.80038381321523 | 4310 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_waist | 1 | 76.21809744779581 | 77.13603273208089 | 78.15173447667419 | 4310 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_waist | 1 | 75.08120649651973 | 76.43609836344807 | 76.63029504408556 | 4310 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_waist | 1 | 75.93967517401393 | 75.4277540793041 | 78.27825014408485 | 4310 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_waist | 1 | 77.70301624129931 | 79.09525756956417 | 79.50060475572067 | 4310 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_waist | 1 | 75.68445475638052 | 77.01243480964862 | 77.21857946976522 | 4310 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_waist | 1 | 77.81902552204176 | 77.52833621899227 | 79.95420636922282 | 4310 | ok |
| unimts | 68.61 | all | 128 | True | False | phone_waist | 1 |  |  |  |  | n/a |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_forearm | 1 | 47.79969650986343 | 34.98631623762692 | 46.61283698213883 | 1977 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_forearm | 1 | 47.496206373292864 | 46.03972605397452 | 52.207185965940674 | 1977 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_forearm | 1 | 47.496206373292864 | 46.03972605397452 | 52.207185965940674 | 1977 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_forearm | 1 | 45.321193727870515 | 44.20708030771672 | 50.691885396739366 | 1977 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_forearm | 1 | 49.92412746585736 | 49.69685961969369 | 54.22417388187818 | 1977 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_forearm | 1 | 51.79564997470916 | 50.74598756902973 | 55.59847685067213 | 1977 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_forearm | 1 | 51.694486595852304 | 50.43162420564278 | 55.92716640732927 | 1977 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_forearm | 1 | 54.3247344461305 | 54.23286842029433 | 58.745997952311924 | 1977 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_forearm | 1 | 53.414264036418814 | 53.51146212689179 | 57.335236501451405 | 1977 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_forearm | 1 | 53.76833586241781 | 53.73610787197863 | 57.96950555178193 | 1977 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_forearm | 1 | 57.00556398583713 | 57.61858397121908 | 62.54218943779608 | 1977 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_forearm | 1 | 57.207890743550834 | 57.1993896774015 | 61.017683764458575 | 1977 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_forearm | 1 | 57.81487101669196 | 57.47473943383918 | 62.006892415619205 | 1977 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_forearm | 1 | 59.48406676783004 | 60.82530814369505 | 65.60286234804549 | 1977 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_forearm | 1 | 58.82650480526049 | 59.01960470538749 | 63.0328512793087 | 1977 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_forearm | 1 | 59.63581183611532 | 59.244619101133054 | 63.85672962250375 | 1977 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_forearm | 1 | 59.22879177377892 | 52.79259131451616 | 60.75938843510843 | 1945 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_forearm | 1 | 59.383033419023135 | 50.50526850146778 | 58.782202628454336 | 1945 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_forearm | 1 | 60.9254498714653 | 51.893278991668815 | 60.936079548456156 | 1945 | ok |
| unimts | 68.61 | all | 64 | True | False | phone_forearm | 1 |  |  |  |  | n/a |
| unimts | 68.61 | all | 128 | True | False | phone_forearm | 1 |  |  |  |  | n/a |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_thigh | 1 | 36.31578947368421 | 25.417457530286804 | 37.731002250170235 | 1710 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_thigh | 1 | 51.75438596491229 | 54.80326252908863 | 58.21912613071335 | 1710 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_thigh | 1 | 51.75438596491229 | 54.80326252908863 | 58.21912613071335 | 1710 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_thigh | 1 | 49.473684210526315 | 52.9039787595208 | 57.771628508653336 | 1710 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_thigh | 1 | 57.60233918128655 | 62.88967710135785 | 65.59148662075289 | 1710 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_thigh | 1 | 53.274853801169584 | 56.74565286896749 | 60.66603839282501 | 1710 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_thigh | 1 | 53.85964912280702 | 57.5721161874625 | 62.46452565276116 | 1710 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_thigh | 1 | 64.32748538011695 | 68.27024161247948 | 70.39418737585943 | 1710 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_thigh | 1 | 57.719298245614034 | 62.67159396692017 | 66.28172270335632 | 1710 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_thigh | 1 | 58.83040935672514 | 63.91654115738332 | 67.65289794942966 | 1710 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_thigh | 1 | 65.6140350877193 | 70.62316439353074 | 72.41601512303365 | 1710 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_thigh | 1 | 60.4093567251462 | 66.6908392390214 | 69.05284098327141 | 1710 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_thigh | 1 | 61.40350877192983 | 67.89935098497327 | 70.05233629553058 | 1710 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_thigh | 1 | 66.60818713450293 | 71.48887099015313 | 72.60482529111209 | 1710 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_thigh | 1 | 61.16959064327485 | 69.09734894809375 | 69.67489848580384 | 1710 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_thigh | 1 | 63.450292397660824 | 71.04445332658887 | 71.86001482231967 | 1710 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_thigh | 1 | 67.30994152046783 | 72.0677755953797 | 73.20799659545631 | 1710 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_thigh | 1 | 64.03508771929825 | 71.57454193393447 | 72.35283153095507 | 1710 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_thigh | 1 | 65.67251461988303 | 72.7092578247615 | 73.90713638007549 | 1710 | ok |
| unimts | 68.61 | all | 64 | True | False | phone_thigh | 1 |  |  |  |  | n/a |
| unimts | 68.61 | all | 128 | True | False | phone_thigh | 1 |  |  |  |  | n/a |
| unimts | 68.61 | zero_support | 0 | True | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| unimts | 68.61 | 1nn | 1 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 55.35372848948374 | 55.252866213492126 | 61.98738364881247 | 1046 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 55.35372848948374 | 55.252866213492126 | 61.98738364881247 | 1046 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 54.20650095602294 | 52.77480012550508 | 60.870171767848305 | 1046 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 61.089866156787764 | 61.34852229978882 | 65.88628468848174 | 1046 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 56.21414913957935 | 56.5541925637221 | 62.12255957999655 | 1046 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 56.21414913957935 | 56.543562970084736 | 62.93604044701064 | 1046 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 66.92160611854685 | 67.52407728957172 | 70.61701942042649 | 1046 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 59.17782026768642 | 61.957271806450265 | 65.27100317289282 | 1046 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 59.56022944550669 | 62.3952049641677 | 65.93342839445211 | 1046 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 69.12045889101339 | 71.45097107638398 | 73.9368234033651 | 1046 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 61.47227533460803 | 65.79763021313634 | 68.08726473149743 | 1046 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 62.23709369024857 | 66.45045633953127 | 69.13086710257352 | 1046 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 72.41715399610136 | 62.35546918276735 | 72.93708205147233 | 1026 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 59.16179337231969 | 52.394909424906935 | 62.3559851733385 | 1026 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 62.47563352826511 | 55.28423742621587 | 65.17591889482175 | 1026 | ok |
| unimts | 68.61 | all | 32 | True | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| unimts | 68.61 | all | 64 | True | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| unimts | 68.61 | all | 128 | True | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_waist | 1 | 14.19953596287703 | 3.183021689957631 | 12.540650406504064 | 4310 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_waist | 1 | 21.62412993039443 | 20.6347079923516 | 21.16218353249287 | 4310 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_waist | 1 | 21.62412993039443 | 20.6347079923516 | 21.16218353249287 | 4310 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_waist | 1 | 21.183294663573086 | 19.685067809550183 | 21.126524212835164 | 4310 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_waist | 1 | 22.621809744779583 | 21.499645196627014 | 22.16401927904027 | 4310 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_waist | 1 | 23.54988399071926 | 22.363023632654862 | 23.17759614518349 | 4310 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_waist | 1 | 23.82830626450116 | 21.33689049139743 | 22.815656280017496 | 4310 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_waist | 1 | 25.08120649651972 | 23.700932863978167 | 24.07881868640122 | 4310 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_waist | 1 | 25.243619489559165 | 23.659177396030838 | 24.235643213541344 | 4310 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_waist | 1 | 24.31554524361949 | 20.564271839932502 | 22.913724632246847 | 4310 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_waist | 1 | 27.16937354988399 | 25.711769908351762 | 25.90676893182811 | 4310 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_waist | 1 | 25.823665893271464 | 24.004215874019334 | 24.488892355769927 | 4310 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_waist | 1 | 25.336426914153133 | 20.980252884422463 | 23.945648830665544 | 4310 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_waist | 1 | 30.580046403712295 | 29.134646307394917 | 29.950217619847034 | 4310 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_waist | 1 | 25.707656612529 | 23.680783331429144 | 24.44504984266352 | 4310 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_waist | 1 | 25.05800464037123 | 19.507638513116014 | 23.410176007845 | 4310 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_waist | 1 | 32.66821345707657 | 31.03156755894784 | 31.517975087078675 | 4310 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_waist | 1 | 26.19489559164733 | 23.8793061060965 | 25.43685920762162 | 4310 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_waist | 1 | 27.447795823665892 | 21.514766978490677 | 25.80303921622794 | 4310 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_waist | 1 | 34.849187935034806 | 33.355633884366895 | 34.441765945129696 | 4310 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_waist | 1 | 27.37819025522042 | 24.556768443003435 | 25.909673082983296 | 4310 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_waist | 1 | 29.721577726218097 | 24.407490437683794 | 28.238233358982313 | 4310 | ok |
| normwear | 1293.86 | all | 128 | True | False | phone_waist | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_forearm | 1 | 16.38846737481032 | 3.5202086049543677 | 12.5 | 1977 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_forearm | 1 | 19.221041982802227 | 16.995022872349335 | 17.457641860898253 | 1977 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_forearm | 1 | 19.221041982802227 | 16.995022872349335 | 17.457641860898253 | 1977 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_forearm | 1 | 18.057663125948405 | 16.294414964435163 | 17.75565758642869 | 1977 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_forearm | 1 | 24.582701062215477 | 22.267360826509368 | 24.147527377559978 | 1977 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_forearm | 1 | 21.699544764795146 | 19.53547930226081 | 20.766332127019556 | 1977 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_forearm | 1 | 17.754172989377846 | 16.16521659428122 | 17.659990396938113 | 1977 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_forearm | 1 | 25.493171471927162 | 23.113279646751494 | 24.977353299463477 | 1977 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_forearm | 1 | 23.520485584218513 | 21.726268369145952 | 24.65553609889492 | 1977 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_forearm | 1 | 21.648963075366716 | 19.851054652608806 | 23.059811573678218 | 1977 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_forearm | 1 | 28.174001011633788 | 25.44793109380899 | 27.18011839252061 | 1977 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_forearm | 1 | 25.03793626707132 | 22.685811581930547 | 24.660839773497035 | 1977 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_forearm | 1 | 23.26757713707638 | 20.037969476071666 | 23.628864386546248 | 1977 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_forearm | 1 | 30.045523520485585 | 26.803516258910477 | 28.51802759854371 | 1977 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_forearm | 1 | 25.341426403641883 | 23.197955961691406 | 27.219397558263164 | 1977 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_forearm | 1 | 27.061203844208396 | 23.06438528364413 | 28.58439264190647 | 1977 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_forearm | 1 | 29.203084832904885 | 25.201015264150016 | 28.620287344885746 | 1945 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_forearm | 1 | 25.655526992287918 | 22.363370240080826 | 25.07320027587977 | 1945 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_forearm | 1 | 27.917737789203084 | 21.29181197742685 | 24.68957998455383 | 1945 | ok |
| normwear | 1293.86 | all | 64 | True | False | phone_forearm | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | all | 128 | True | False | phone_forearm | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_thigh | 1 | 19.005847953216374 | 3.9945919370698126 | 12.5 | 1710 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_thigh | 1 | 21.46198830409357 | 20.07198770044085 | 22.504551875643603 | 1710 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_thigh | 1 | 21.46198830409357 | 20.07198770044085 | 22.504551875643603 | 1710 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_thigh | 1 | 19.5906432748538 | 17.996679727870596 | 20.134355885722876 | 1710 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_thigh | 1 | 24.09356725146199 | 22.21512550952942 | 24.447954559581657 | 1710 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_thigh | 1 | 25.321637426900583 | 23.42106444325957 | 25.73328587586915 | 1710 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_thigh | 1 | 21.34502923976608 | 19.87948704427455 | 22.609716143061124 | 1710 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_thigh | 1 | 25.6140350877193 | 24.04694842339391 | 26.38005421346049 | 1710 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_thigh | 1 | 24.736842105263158 | 22.865526775006312 | 25.064984926260657 | 1710 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_thigh | 1 | 22.45614035087719 | 20.640639630808703 | 24.218320867717658 | 1710 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_thigh | 1 | 29.064327485380115 | 26.73500929030186 | 28.49652518728241 | 1710 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_thigh | 1 | 25.2046783625731 | 23.395228752257662 | 26.19110051887668 | 1710 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_thigh | 1 | 23.04093567251462 | 20.856595497320292 | 24.670380371377515 | 1710 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_thigh | 1 | 32.514619883040936 | 30.861841646392485 | 33.13281867047656 | 1710 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_thigh | 1 | 27.134502923976605 | 24.984259598608627 | 28.141924762588978 | 1710 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_thigh | 1 | 25.672514619883042 | 23.07670983632183 | 29.820873789696655 | 1710 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_thigh | 1 | 36.78362573099415 | 34.69545295475768 | 37.292974061305515 | 1710 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_thigh | 1 | 27.77777777777778 | 26.12468152658866 | 30.77338619632987 | 1710 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_thigh | 1 | 27.66081871345029 | 25.442624923143132 | 33.898724618381785 | 1710 | ok |
| normwear | 1293.86 | all | 64 | True | False | phone_thigh | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | all | 128 | True | False | phone_thigh | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | zero_support | 0 | True | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 23.900573613766728 | 20.80521364971048 | 21.921529332129563 | 1046 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 23.900573613766728 | 20.80521364971048 | 21.921529332129563 | 1046 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 19.21606118546845 | 16.433388268682986 | 17.559993530521066 | 1046 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 25.812619502868067 | 22.1711510032503 | 23.285781818308955 | 1046 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 25.430210325047803 | 22.329344450966982 | 24.39027657101378 | 1046 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 22.753346080305928 | 18.683540647001205 | 20.61540052079974 | 1046 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 27.820267686424476 | 23.83060654744234 | 25.535121841963715 | 1046 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 26.38623326959847 | 22.71722645608558 | 23.76080427479931 | 1046 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 24.18738049713193 | 19.636984049875394 | 23.548305145181736 | 1046 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 35.37284894837477 | 30.857016728951002 | 33.71247362522959 | 1046 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 25.334608030592737 | 21.791007099304853 | 23.86498353141335 | 1046 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 25.1434034416826 | 20.474307808949273 | 26.661360949067188 | 1046 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 36.35477582846004 | 30.004439938503584 | 34.96534385064016 | 1026 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 26.510721247563353 | 21.440895001378408 | 26.122226258355553 | 1026 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_forearm+phone_thigh+phone_waist | 3 | 27.87524366471735 | 19.374693919477714 | 28.75726279318009 | 1026 | ok |
| normwear | 1293.86 | all | 32 | True | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| normwear | 1293.86 | all | 64 | True | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| normwear | 1293.86 | all | 128 | True | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 1 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 2 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 4 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 8 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 16 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 32 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 64 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 128 | False | False | phone_waist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 1 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 2 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 4 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 8 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 16 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 32 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 64 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 128 | False | False | phone_forearm | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 1 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 2 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 4 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 8 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 16 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 32 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 64 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 128 | False | False | phone_thigh | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | zero_support | 0 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 1 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 2 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 4 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 8 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 16 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 32 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 64 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |
| limubert_x | 0.055 | all | 128 | False | False | phone_forearm+phone_thigh+phone_waist | 3 |  |  |  |  | n/a |

### shoaib
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | stream | n_devices | accuracy | f1_macro | balanced_accuracy | n_queries | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halo | 2.203 | halo-classifier | 0 | True | True | phone_right_pocket | 1 | 71.30952380952381 | 68.96929632284827 | 71.3095238095238 | 840 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_right_pocket | 1 | 59.523809523809526 | 56.21163600761151 | 59.52380952380951 | 840 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_right_pocket | 1 | 75.95238095238095 | 76.27485141105383 | 75.95238095238096 | 840 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_right_pocket | 1 | 75.95238095238095 | 76.27485141105383 | 75.95238095238096 | 840 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_right_pocket | 1 | 76.66666666666667 | 76.6335144726971 | 76.66666666666666 | 840 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_right_pocket | 1 | 79.76190476190477 | 79.27779383681872 | 79.76190476190476 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_right_pocket | 1 | 75.95238095238095 | 76.21013250525 | 75.95238095238096 | 840 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_right_pocket | 1 | 85.0 | 85.0768295797744 | 85.0 | 840 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_right_pocket | 1 | 85.71428571428571 | 85.72593109691418 | 85.71428571428571 | 840 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_right_pocket | 1 | 86.66666666666667 | 86.51672768536241 | 86.66666666666664 | 840 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_right_pocket | 1 | 83.69047619047619 | 83.38013987982762 | 83.69047619047619 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_right_pocket | 1 | 86.30952380952381 | 86.24933057128767 | 86.3095238095238 | 840 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_right_pocket | 1 | 88.57142857142857 | 88.64469813631783 | 88.57142857142858 | 840 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_right_pocket | 1 | 87.97619047619048 | 88.07415687188065 | 87.97619047619048 | 840 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_right_pocket | 1 | 90.23809523809524 | 90.16738892524133 | 90.23809523809524 | 840 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_right_pocket | 1 | 87.26190476190476 | 86.9777198058429 | 87.26190476190476 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_right_pocket | 1 | 89.40476190476191 | 89.43636812239329 | 89.4047619047619 | 840 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_right_pocket | 1 | 89.76190476190476 | 89.7150810430234 | 89.76190476190477 | 840 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_right_pocket | 1 | 90.11904761904762 | 90.21103695223871 | 90.11904761904762 | 840 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_right_pocket | 1 | 92.14285714285714 | 92.07575092411079 | 92.14285714285715 | 840 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_right_pocket | 1 | 87.85714285714286 | 87.60884500189596 | 87.85714285714285 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_right_pocket | 1 | 90.35714285714286 | 90.23112245086253 | 90.35714285714288 | 840 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_right_pocket | 1 | 91.07142857142857 | 91.03247529509865 | 91.07142857142857 | 840 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_right_pocket | 1 | 91.30952380952381 | 91.35464282213809 | 91.30952380952382 | 840 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_right_pocket | 1 | 94.04761904761905 | 93.98494860801861 | 94.04761904761905 | 840 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_right_pocket | 1 | 89.88095238095238 | 89.75561904108832 | 89.88095238095238 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_right_pocket | 1 | 91.07142857142857 | 90.9432718604769 | 91.07142857142857 | 840 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_right_pocket | 1 | 91.66666666666666 | 91.52799122110606 | 91.66666666666667 | 840 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_right_pocket | 1 | 91.42857142857143 | 91.53163253593398 | 91.42857142857143 | 840 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_right_pocket | 1 | 95.35714285714286 | 95.33195720983161 | 95.35714285714285 | 840 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_right_pocket | 1 | 89.76190476190476 | 89.53948501784262 | 89.76190476190474 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_right_pocket | 1 | 91.78571428571428 | 91.64388740599338 | 91.78571428571428 | 840 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_right_pocket | 1 | 93.33333333333333 | 93.29145513627137 | 93.33333333333333 | 840 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_right_pocket | 1 | 91.42857142857143 | 91.55188292598608 | 91.42857142857143 | 840 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_right_pocket | 1 | 96.78571428571429 | 96.75600434693344 | 96.78571428571428 | 840 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_right_pocket | 1 | 90.5952380952381 | 90.47751318547844 | 90.5952380952381 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_right_pocket | 1 | 91.9047619047619 | 91.81092650906182 | 91.9047619047619 | 840 | ok |
| halo | 0.789 | all | 128 | True | True | phone_right_pocket | 1 |  |  |  |  | n/a |
| halo | 2.203 | halo-classifier | 0 | True | True | phone_left_pocket | 1 | 75.83333333333333 | 73.61961560324538 | 75.83333333333334 | 840 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_left_pocket | 1 | 60.71428571428571 | 58.74762252143485 | 60.71428571428573 | 840 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_left_pocket | 1 | 77.5 | 77.70767492212391 | 77.5 | 840 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_left_pocket | 1 | 77.5 | 77.70767492212391 | 77.5 | 840 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_left_pocket | 1 | 79.76190476190477 | 79.51368839288259 | 79.76190476190476 | 840 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_left_pocket | 1 | 82.85714285714286 | 82.4728055400162 | 82.85714285714285 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_left_pocket | 1 | 77.5 | 77.50919227289802 | 77.5 | 840 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_left_pocket | 1 | 82.97619047619048 | 83.03846101928701 | 82.9761904761905 | 840 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_left_pocket | 1 | 85.35714285714285 | 85.3664781822291 | 85.35714285714285 | 840 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_left_pocket | 1 | 86.42857142857143 | 86.12781183121947 | 86.42857142857142 | 840 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_left_pocket | 1 | 85.47619047619047 | 84.83551810515046 | 85.47619047619047 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_left_pocket | 1 | 84.04761904761905 | 83.94198516791445 | 84.04761904761904 | 840 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_left_pocket | 1 | 87.26190476190476 | 87.28165452676265 | 87.26190476190477 | 840 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_left_pocket | 1 | 86.90476190476191 | 86.91547949705354 | 86.9047619047619 | 840 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_left_pocket | 1 | 90.5952380952381 | 90.4237667068399 | 90.5952380952381 | 840 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_left_pocket | 1 | 86.78571428571429 | 86.06002871356802 | 86.78571428571429 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_left_pocket | 1 | 88.33333333333333 | 88.24081509882978 | 88.33333333333334 | 840 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_left_pocket | 1 | 90.95238095238095 | 90.91992053242842 | 90.95238095238096 | 840 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_left_pocket | 1 | 90.0 | 90.01059801279673 | 90.0 | 840 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_left_pocket | 1 | 91.30952380952381 | 91.17796860510923 | 91.30952380952381 | 840 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_left_pocket | 1 | 89.16666666666667 | 88.81720125072673 | 89.16666666666666 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_left_pocket | 1 | 91.78571428571428 | 91.66240038711383 | 91.78571428571428 | 840 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_left_pocket | 1 | 91.66666666666666 | 91.68427391151342 | 91.66666666666667 | 840 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_left_pocket | 1 | 91.07142857142857 | 91.07817928966855 | 91.07142857142857 | 840 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_left_pocket | 1 | 93.45238095238095 | 93.40633967491058 | 93.45238095238095 | 840 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_left_pocket | 1 | 89.16666666666667 | 88.69628628356217 | 89.16666666666666 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_left_pocket | 1 | 92.26190476190477 | 92.1961281673079 | 92.26190476190476 | 840 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_left_pocket | 1 | 92.5 | 92.49666896460225 | 92.5 | 840 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_left_pocket | 1 | 91.30952380952381 | 91.29647901022476 | 91.30952380952381 | 840 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_left_pocket | 1 | 93.92857142857143 | 93.8994532704222 | 93.92857142857143 | 840 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_left_pocket | 1 | 89.76190476190476 | 89.40422723998196 | 89.76190476190474 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_left_pocket | 1 | 92.02380952380952 | 91.96773566538896 | 92.02380952380952 | 840 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_left_pocket | 1 | 94.28571428571428 | 94.27097837234184 | 94.28571428571429 | 840 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_left_pocket | 1 | 91.66666666666666 | 91.68479194465472 | 91.66666666666667 | 840 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_left_pocket | 1 | 95.83333333333334 | 95.82247443277004 | 95.83333333333333 | 840 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_left_pocket | 1 | 90.11904761904762 | 89.79739425179774 | 90.1190476190476 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_left_pocket | 1 | 92.85714285714286 | 92.80780895543866 | 92.85714285714286 | 840 | ok |
| halo | 0.789 | all | 128 | True | True | phone_left_pocket | 1 |  |  |  |  | n/a |
| halo | 2.203 | halo-classifier | 0 | True | True | phone_belt | 1 | 74.40476190476191 | 73.96473698468219 | 74.4047619047619 | 840 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_belt | 1 | 57.02380952380952 | 52.91744517652738 | 57.02380952380953 | 840 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_belt | 1 | 89.52380952380953 | 89.61943998703336 | 89.52380952380953 | 840 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_belt | 1 | 89.52380952380953 | 89.61943998703336 | 89.52380952380953 | 840 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_belt | 1 | 91.42857142857143 | 91.4020880349117 | 91.42857142857143 | 840 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_belt | 1 | 92.85714285714286 | 92.89846604496968 | 92.85714285714286 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_belt | 1 | 88.92857142857142 | 88.92878212757529 | 88.92857142857142 | 840 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_belt | 1 | 90.95238095238095 | 90.99477153571779 | 90.95238095238096 | 840 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_belt | 1 | 92.38095238095238 | 92.40530918552989 | 92.38095238095238 | 840 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_belt | 1 | 91.54761904761905 | 91.43746707676938 | 91.54761904761905 | 840 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_belt | 1 | 92.5 | 92.55762053444526 | 92.5 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_belt | 1 | 90.83333333333333 | 90.75216723033078 | 90.83333333333333 | 840 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_belt | 1 | 93.0952380952381 | 93.14242814825116 | 93.0952380952381 | 840 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_belt | 1 | 94.4047619047619 | 94.42229984530273 | 94.4047619047619 | 840 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_belt | 1 | 93.92857142857143 | 93.92026586053427 | 93.92857142857142 | 840 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_belt | 1 | 92.97619047619048 | 93.01768951748528 | 92.97619047619047 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_belt | 1 | 93.80952380952381 | 93.79783137507006 | 93.80952380952381 | 840 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_belt | 1 | 95.35714285714286 | 95.37668731974854 | 95.35714285714285 | 840 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_belt | 1 | 95.35714285714286 | 95.36986118340752 | 95.35714285714286 | 840 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_belt | 1 | 95.35714285714286 | 95.37502427986571 | 95.35714285714285 | 840 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_belt | 1 | 94.4047619047619 | 94.44646097915468 | 94.4047619047619 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_belt | 1 | 95.5952380952381 | 95.586251437863 | 95.59523809523809 | 840 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_belt | 1 | 96.07142857142857 | 96.08904159557596 | 96.07142857142857 | 840 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_belt | 1 | 96.78571428571429 | 96.78258830797887 | 96.78571428571429 | 840 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_belt | 1 | 97.02380952380952 | 97.02033479887275 | 97.02380952380955 | 840 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_belt | 1 | 94.88095238095238 | 94.91220926362828 | 94.88095238095238 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_belt | 1 | 97.14285714285714 | 97.14161826023722 | 97.14285714285712 | 840 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_belt | 1 | 96.66666666666667 | 96.67968239894488 | 96.66666666666667 | 840 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_belt | 1 | 96.78571428571429 | 96.78929631753206 | 96.78571428571429 | 840 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_belt | 1 | 97.38095238095238 | 97.3860262533494 | 97.38095238095238 | 840 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_belt | 1 | 94.88095238095238 | 94.89971769723405 | 94.88095238095238 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_belt | 1 | 97.85714285714285 | 97.8580211423712 | 97.85714285714285 | 840 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_belt | 1 | 96.30952380952381 | 96.32418920536597 | 96.3095238095238 | 840 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_belt | 1 | 97.14285714285714 | 97.14255507713509 | 97.14285714285714 | 840 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_belt | 1 | 98.21428571428571 | 98.21328572989162 | 98.21428571428571 | 840 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_belt | 1 | 95.47619047619048 | 95.49168345713677 | 95.47619047619048 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_belt | 1 | 97.85714285714285 | 97.85618580086765 | 97.85714285714285 | 840 | ok |
| halo | 0.789 | all | 128 | True | True | phone_belt | 1 |  |  |  |  | n/a |
| halo | 2.203 | halo-classifier | 0 | True | True | watch_wrist_proxy | 1 | 60.71428571428571 | 57.33047724184159 | 60.71428571428571 | 840 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | watch_wrist_proxy | 1 | 43.57142857142857 | 36.443591265024025 | 43.57142857142857 | 840 | ok |
| halo | 0.789 | 1nn | 1 | True | True | watch_wrist_proxy | 1 | 81.42857142857143 | 81.33760609317156 | 81.42857142857143 | 840 | ok |
| halo | 0.789 | prototype | 1 | True | True | watch_wrist_proxy | 1 | 81.42857142857143 | 81.33760609317156 | 81.42857142857143 | 840 | ok |
| halo | 0.789 | ridge | 1 | True | True | watch_wrist_proxy | 1 | 80.83333333333333 | 80.68454632227325 | 80.83333333333333 | 840 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | watch_wrist_proxy | 1 | 81.19047619047619 | 80.87848535346876 | 81.19047619047619 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | watch_wrist_proxy | 1 | 81.07142857142857 | 80.8863960602703 | 81.07142857142858 | 840 | ok |
| halo | 0.789 | 1nn | 2 | True | True | watch_wrist_proxy | 1 | 82.97619047619048 | 82.86945626482625 | 82.97619047619048 | 840 | ok |
| halo | 0.789 | prototype | 2 | True | True | watch_wrist_proxy | 1 | 85.83333333333333 | 85.639746134012 | 85.83333333333333 | 840 | ok |
| halo | 0.789 | ridge | 2 | True | True | watch_wrist_proxy | 1 | 86.90476190476191 | 86.59122026203434 | 86.90476190476191 | 840 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | watch_wrist_proxy | 1 | 85.47619047619047 | 85.34996593828973 | 85.47619047619047 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | watch_wrist_proxy | 1 | 84.16666666666667 | 84.03714850751905 | 84.16666666666667 | 840 | ok |
| halo | 0.789 | 1nn | 4 | True | True | watch_wrist_proxy | 1 | 86.78571428571429 | 86.79355505651436 | 86.78571428571429 | 840 | ok |
| halo | 0.789 | prototype | 4 | True | True | watch_wrist_proxy | 1 | 86.90476190476191 | 86.7933603996017 | 86.9047619047619 | 840 | ok |
| halo | 0.789 | ridge | 4 | True | True | watch_wrist_proxy | 1 | 87.85714285714286 | 87.5162177312317 | 87.85714285714286 | 840 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | watch_wrist_proxy | 1 | 87.02380952380952 | 87.17674245945936 | 87.02380952380952 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | watch_wrist_proxy | 1 | 87.14285714285714 | 87.15022549194293 | 87.14285714285715 | 840 | ok |
| halo | 0.789 | 1nn | 8 | True | True | watch_wrist_proxy | 1 | 87.14285714285714 | 87.07436220544999 | 87.14285714285714 | 840 | ok |
| halo | 0.789 | prototype | 8 | True | True | watch_wrist_proxy | 1 | 89.04761904761904 | 88.9589841560568 | 89.04761904761904 | 840 | ok |
| halo | 0.789 | ridge | 8 | True | True | watch_wrist_proxy | 1 | 89.52380952380953 | 89.22978870758679 | 89.52380952380953 | 840 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | watch_wrist_proxy | 1 | 86.78571428571429 | 86.8067100110982 | 86.78571428571428 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | watch_wrist_proxy | 1 | 88.33333333333333 | 88.30474261086104 | 88.33333333333334 | 840 | ok |
| halo | 0.789 | 1nn | 16 | True | True | watch_wrist_proxy | 1 | 87.14285714285714 | 87.04070434867633 | 87.14285714285715 | 840 | ok |
| halo | 0.789 | prototype | 16 | True | True | watch_wrist_proxy | 1 | 89.88095238095238 | 89.70609651860453 | 89.88095238095238 | 840 | ok |
| halo | 0.789 | ridge | 16 | True | True | watch_wrist_proxy | 1 | 90.35714285714286 | 90.08102313091308 | 90.35714285714286 | 840 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | watch_wrist_proxy | 1 | 87.73809523809524 | 87.70979138798184 | 87.73809523809524 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | watch_wrist_proxy | 1 | 88.33333333333333 | 88.19558967215329 | 88.33333333333334 | 840 | ok |
| halo | 0.789 | 1nn | 32 | True | True | watch_wrist_proxy | 1 | 87.73809523809524 | 87.72869380858876 | 87.73809523809526 | 840 | ok |
| halo | 0.789 | prototype | 32 | True | True | watch_wrist_proxy | 1 | 90.35714285714286 | 90.22591757251546 | 90.35714285714286 | 840 | ok |
| halo | 0.789 | ridge | 32 | True | True | watch_wrist_proxy | 1 | 91.42857142857143 | 91.2222468511371 | 91.42857142857143 | 840 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | watch_wrist_proxy | 1 | 88.45238095238095 | 88.49972699991808 | 88.45238095238095 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | watch_wrist_proxy | 1 | 89.76190476190476 | 89.78526710889193 | 89.76190476190476 | 840 | ok |
| halo | 0.789 | 1nn | 64 | True | True | watch_wrist_proxy | 1 | 87.73809523809524 | 87.6296685642269 | 87.73809523809524 | 840 | ok |
| halo | 0.789 | prototype | 64 | True | True | watch_wrist_proxy | 1 | 90.95238095238095 | 90.85530450361927 | 90.95238095238096 | 840 | ok |
| halo | 0.789 | ridge | 64 | True | True | watch_wrist_proxy | 1 | 92.61904761904762 | 92.5161779194694 | 92.61904761904762 | 840 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | watch_wrist_proxy | 1 | 88.69047619047619 | 88.75584525501277 | 88.6904761904762 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | watch_wrist_proxy | 1 | 90.71428571428571 | 90.7438001820765 | 90.71428571428572 | 840 | ok |
| halo | 0.789 | all | 128 | True | True | watch_wrist_proxy | 1 |  |  |  |  | n/a |
| halo | 2.203 | halo-classifier | 0 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 74.52380952380952 | 71.07157372749866 | 74.52380952380953 | 840 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| halo | 0.789 | 1nn | 1 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 88.33333333333333 | 88.34430206283682 | 88.33333333333334 | 840 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 88.33333333333333 | 88.34430206283682 | 88.33333333333334 | 840 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 88.69047619047619 | 88.62313473174223 | 88.69047619047619 | 840 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 87.61904761904762 | 87.48494370557563 | 87.61904761904763 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 87.38095238095238 | 87.37353692402931 | 87.3809523809524 | 840 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 93.45238095238095 | 93.48629536176514 | 93.45238095238095 | 840 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 93.69047619047619 | 93.68344556234237 | 93.69047619047619 | 840 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 94.52380952380952 | 94.44935855924747 | 94.52380952380952 | 840 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 92.02380952380952 | 91.97123095729091 | 92.02380952380953 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 93.92857142857143 | 93.93878595847285 | 93.92857142857143 | 840 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 95.35714285714286 | 95.35087907597631 | 95.35714285714285 | 840 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 95.0 | 94.97039005112194 | 95.0 | 840 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 96.30952380952381 | 96.26062452738495 | 96.3095238095238 | 840 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 91.9047619047619 | 91.84728106823626 | 91.90476190476191 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 95.5952380952381 | 95.56538781492755 | 95.59523809523809 | 840 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 95.23809523809523 | 95.22815875689535 | 95.23809523809524 | 840 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 96.19047619047619 | 96.18165880819433 | 96.19047619047619 | 840 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 97.26190476190476 | 97.24838949428994 | 97.26190476190476 | 840 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 93.80952380952381 | 93.83078766930693 | 93.80952380952382 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 96.42857142857143 | 96.41257474624634 | 96.42857142857143 | 840 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 97.26190476190476 | 97.24556759683975 | 97.26190476190476 | 840 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 96.54761904761905 | 96.51924744240826 | 96.54761904761904 | 840 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 97.73809523809524 | 97.72071749909472 | 97.73809523809524 | 840 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 94.64285714285714 | 94.66287861033402 | 94.64285714285714 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 97.61904761904762 | 97.59908735001433 | 97.61904761904762 | 840 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 97.85714285714285 | 97.85657209810087 | 97.85714285714286 | 840 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 96.54761904761905 | 96.52362743047759 | 96.54761904761907 | 840 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 98.09523809523809 | 98.08651926032617 | 98.0952380952381 | 840 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 94.52380952380952 | 94.54963210838442 | 94.52380952380953 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 97.73809523809524 | 97.71620353095138 | 97.73809523809524 | 840 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 97.5 | 97.48895213326763 | 97.5 | 840 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 96.9047619047619 | 96.88627059424365 | 96.90476190476191 | 840 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 98.57142857142858 | 98.56851871735645 | 98.57142857142858 | 840 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 95.11904761904762 | 95.14253109148612 | 95.11904761904762 | 840 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 97.85714285714285 | 97.83783368003178 | 97.85714285714285 | 840 | ok |
| halo | 0.789 | all | 128 | True | True | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_right_pocket | 1 | 46.19047619047619 | 42.77487819103753 | 46.19047619047619 | 840 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_right_pocket | 1 | 67.97619047619048 | 67.50885832043215 | 67.97619047619048 | 840 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_right_pocket | 1 | 67.97619047619048 | 67.50885832043215 | 67.97619047619048 | 840 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_right_pocket | 1 | 67.85714285714286 | 67.22798209729814 | 67.85714285714286 | 840 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_right_pocket | 1 | 71.19047619047619 | 70.758165647616 | 71.19047619047619 | 840 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_right_pocket | 1 | 70.11904761904762 | 69.4776203946235 | 70.11904761904762 | 840 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_right_pocket | 1 | 70.35714285714286 | 69.4107485467416 | 70.35714285714285 | 840 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_right_pocket | 1 | 77.73809523809524 | 77.42177014402003 | 77.73809523809526 | 840 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_right_pocket | 1 | 75.95238095238095 | 75.38696150690375 | 75.95238095238095 | 840 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_right_pocket | 1 | 76.07142857142857 | 75.1996379575471 | 76.07142857142858 | 840 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_right_pocket | 1 | 82.5 | 82.38599856242122 | 82.5 | 840 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_right_pocket | 1 | 80.11904761904762 | 79.80815653308504 | 80.1190476190476 | 840 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_right_pocket | 1 | 80.95238095238095 | 80.52451107678716 | 80.95238095238096 | 840 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_right_pocket | 1 | 85.23809523809524 | 85.28809188239585 | 85.23809523809523 | 840 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_right_pocket | 1 | 80.0 | 79.68468076081898 | 80.0 | 840 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_right_pocket | 1 | 84.28571428571429 | 84.01313630040111 | 84.28571428571429 | 840 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_right_pocket | 1 | 88.92857142857142 | 88.9089011277592 | 88.92857142857142 | 840 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_right_pocket | 1 | 80.5952380952381 | 80.34867260117923 | 80.5952380952381 | 840 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_right_pocket | 1 | 87.5 | 87.34672086159547 | 87.5 | 840 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_right_pocket | 1 | 88.92857142857142 | 88.92510009141878 | 88.92857142857143 | 840 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_right_pocket | 1 | 79.52380952380952 | 79.24726750589774 | 79.52380952380952 | 840 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_right_pocket | 1 | 88.80952380952381 | 88.66883729021046 | 88.80952380952381 | 840 | ok |
| harnet | 4.49 | all | 128 | False | False | phone_right_pocket | 1 |  |  |  |  | n/a |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_left_pocket | 1 | 49.28571428571429 | 45.53853831463797 | 49.28571428571429 | 840 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_left_pocket | 1 | 62.857142857142854 | 62.20838010435654 | 62.85714285714287 | 840 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_left_pocket | 1 | 62.857142857142854 | 62.20838010435654 | 62.85714285714287 | 840 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_left_pocket | 1 | 65.5952380952381 | 64.79858211274862 | 65.5952380952381 | 840 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_left_pocket | 1 | 74.64285714285714 | 74.34882494664747 | 74.64285714285714 | 840 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_left_pocket | 1 | 73.69047619047619 | 73.43997432755522 | 73.6904761904762 | 840 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_left_pocket | 1 | 75.0 | 74.5511559887017 | 75.0 | 840 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_left_pocket | 1 | 80.11904761904762 | 79.86366925893097 | 80.11904761904762 | 840 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_left_pocket | 1 | 76.42857142857142 | 75.96604424042272 | 76.42857142857142 | 840 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_left_pocket | 1 | 77.73809523809524 | 77.22699608501472 | 77.73809523809524 | 840 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_left_pocket | 1 | 84.28571428571429 | 84.13976962009902 | 84.28571428571429 | 840 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_left_pocket | 1 | 80.5952380952381 | 80.34009294186266 | 80.5952380952381 | 840 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_left_pocket | 1 | 82.97619047619048 | 82.70733481160478 | 82.97619047619048 | 840 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_left_pocket | 1 | 88.92857142857142 | 88.85239759591967 | 88.92857142857142 | 840 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_left_pocket | 1 | 81.42857142857143 | 81.33554366756746 | 81.42857142857143 | 840 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_left_pocket | 1 | 84.16666666666667 | 84.06660585550149 | 84.16666666666669 | 840 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_left_pocket | 1 | 89.88095238095238 | 89.7846976934604 | 89.88095238095238 | 840 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_left_pocket | 1 | 80.71428571428572 | 80.56157512377713 | 80.71428571428572 | 840 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_left_pocket | 1 | 84.88095238095238 | 84.59323113462693 | 84.8809523809524 | 840 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_left_pocket | 1 | 91.66666666666666 | 91.60799721021417 | 91.66666666666666 | 840 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_left_pocket | 1 | 83.0952380952381 | 82.99272325739227 | 83.09523809523809 | 840 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_left_pocket | 1 | 88.69047619047619 | 88.51999354344994 | 88.69047619047619 | 840 | ok |
| harnet | 4.49 | all | 128 | False | False | phone_left_pocket | 1 |  |  |  |  | n/a |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_belt | 1 | 50.95238095238095 | 49.377927007574584 | 50.95238095238096 | 840 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_belt | 1 | 64.28571428571429 | 63.75751181928918 | 64.28571428571429 | 840 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_belt | 1 | 64.28571428571429 | 63.75751181928918 | 64.28571428571429 | 840 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_belt | 1 | 61.54761904761905 | 60.5700538235495 | 61.54761904761905 | 840 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_belt | 1 | 68.69047619047619 | 68.2704919599255 | 68.69047619047619 | 840 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_belt | 1 | 68.80952380952381 | 68.38099555778876 | 68.80952380952381 | 840 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_belt | 1 | 66.54761904761905 | 65.69582618928348 | 66.54761904761905 | 840 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_belt | 1 | 76.07142857142857 | 75.91229257764222 | 76.07142857142858 | 840 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_belt | 1 | 74.16666666666667 | 73.83626754791977 | 74.16666666666669 | 840 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_belt | 1 | 73.45238095238096 | 73.04006383482572 | 73.45238095238095 | 840 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_belt | 1 | 79.76190476190477 | 79.69617183084098 | 79.76190476190477 | 840 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_belt | 1 | 76.54761904761904 | 76.31609602206798 | 76.54761904761904 | 840 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_belt | 1 | 80.0 | 79.8010698213635 | 79.99999999999999 | 840 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_belt | 1 | 82.73809523809523 | 82.70953875058864 | 82.73809523809523 | 840 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_belt | 1 | 79.04761904761905 | 78.92296978238856 | 79.04761904761905 | 840 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_belt | 1 | 84.04761904761905 | 83.9370672948505 | 84.04761904761904 | 840 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_belt | 1 | 86.42857142857143 | 86.39493355138147 | 86.42857142857142 | 840 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_belt | 1 | 80.11904761904762 | 80.01839090068746 | 80.1190476190476 | 840 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_belt | 1 | 86.30952380952381 | 86.20480391949037 | 86.30952380952381 | 840 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_belt | 1 | 86.66666666666667 | 86.63810550547811 | 86.66666666666669 | 840 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_belt | 1 | 82.26190476190476 | 82.18275333098427 | 82.26190476190476 | 840 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_belt | 1 | 89.28571428571429 | 89.22564375266369 | 89.28571428571426 | 840 | ok |
| harnet | 4.49 | all | 128 | False | False | phone_belt | 1 |  |  |  |  | n/a |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | watch_wrist_proxy | 1 | 59.76190476190476 | 57.5291073917519 | 59.76190476190476 | 840 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | watch_wrist_proxy | 1 | 69.52380952380952 | 69.06140522530396 | 69.52380952380953 | 840 | ok |
| harnet | 4.49 | prototype | 1 | False | False | watch_wrist_proxy | 1 | 69.52380952380952 | 69.06140522530396 | 69.52380952380953 | 840 | ok |
| harnet | 4.49 | ridge | 1 | False | False | watch_wrist_proxy | 1 | 68.0952380952381 | 67.26293597340313 | 68.0952380952381 | 840 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | watch_wrist_proxy | 1 | 75.23809523809524 | 75.04857208755374 | 75.23809523809524 | 840 | ok |
| harnet | 4.49 | prototype | 2 | False | False | watch_wrist_proxy | 1 | 74.16666666666667 | 73.9015151234046 | 74.16666666666666 | 840 | ok |
| harnet | 4.49 | ridge | 2 | False | False | watch_wrist_proxy | 1 | 75.0 | 74.47059654336624 | 75.0 | 840 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | watch_wrist_proxy | 1 | 77.73809523809524 | 77.47133493548272 | 77.73809523809524 | 840 | ok |
| harnet | 4.49 | prototype | 4 | False | False | watch_wrist_proxy | 1 | 78.80952380952381 | 78.65799060326347 | 78.80952380952381 | 840 | ok |
| harnet | 4.49 | ridge | 4 | False | False | watch_wrist_proxy | 1 | 78.33333333333333 | 77.92574687035086 | 78.33333333333333 | 840 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | watch_wrist_proxy | 1 | 80.35714285714286 | 80.16858512559402 | 80.35714285714286 | 840 | ok |
| harnet | 4.49 | prototype | 8 | False | False | watch_wrist_proxy | 1 | 80.5952380952381 | 80.4306740091782 | 80.5952380952381 | 840 | ok |
| harnet | 4.49 | ridge | 8 | False | False | watch_wrist_proxy | 1 | 80.5952380952381 | 80.19087835104524 | 80.5952380952381 | 840 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | watch_wrist_proxy | 1 | 80.71428571428572 | 80.65008412978464 | 80.71428571428572 | 840 | ok |
| harnet | 4.49 | prototype | 16 | False | False | watch_wrist_proxy | 1 | 81.19047619047619 | 81.09018525313881 | 81.19047619047618 | 840 | ok |
| harnet | 4.49 | ridge | 16 | False | False | watch_wrist_proxy | 1 | 82.14285714285714 | 81.69928877842318 | 82.14285714285714 | 840 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | watch_wrist_proxy | 1 | 83.0952380952381 | 83.02671401281418 | 83.09523809523809 | 840 | ok |
| harnet | 4.49 | prototype | 32 | False | False | watch_wrist_proxy | 1 | 82.5 | 82.42288985452649 | 82.5 | 840 | ok |
| harnet | 4.49 | ridge | 32 | False | False | watch_wrist_proxy | 1 | 84.76190476190476 | 84.58587335021471 | 84.76190476190474 | 840 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | watch_wrist_proxy | 1 | 84.88095238095238 | 84.91487925844156 | 84.88095238095238 | 840 | ok |
| harnet | 4.49 | prototype | 64 | False | False | watch_wrist_proxy | 1 | 82.5 | 82.40496688455409 | 82.5 | 840 | ok |
| harnet | 4.49 | ridge | 64 | False | False | watch_wrist_proxy | 1 | 87.14285714285714 | 87.00334057456574 | 87.14285714285714 | 840 | ok |
| harnet | 4.49 | all | 128 | False | False | watch_wrist_proxy | 1 |  |  |  |  | n/a |
| harnet | 4.49 | zero_support | 0 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| harnet | 4.49 | 1nn | 1 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 69.76190476190476 | 69.12255922901224 | 69.76190476190476 | 840 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 69.76190476190476 | 69.12255922901224 | 69.76190476190476 | 840 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 66.9047619047619 | 65.97327547319222 | 66.90476190476191 | 840 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 78.69047619047619 | 78.49051485091249 | 78.69047619047619 | 840 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 79.28571428571428 | 79.04344926541106 | 79.28571428571428 | 840 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 74.88095238095238 | 74.46609428033577 | 74.8809523809524 | 840 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 81.19047619047619 | 80.98196626141726 | 81.19047619047619 | 840 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 80.47619047619048 | 80.28556610417058 | 80.47619047619047 | 840 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 78.45238095238095 | 78.13369498014151 | 78.45238095238095 | 840 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 84.76190476190476 | 84.55733143325736 | 84.76190476190476 | 840 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 80.47619047619048 | 80.40928215878301 | 80.47619047619048 | 840 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 81.07142857142857 | 80.96076444700405 | 81.07142857142858 | 840 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 87.02380952380952 | 86.89701264736514 | 87.02380952380952 | 840 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 82.5 | 82.40227313425463 | 82.5 | 840 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 83.0952380952381 | 82.9002094650609 | 83.09523809523809 | 840 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 87.97619047619048 | 87.84679082483956 | 87.97619047619048 | 840 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 83.0952380952381 | 82.98238967511163 | 83.0952380952381 | 840 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 85.11904761904762 | 84.87303410361699 | 85.11904761904762 | 840 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 89.04761904761904 | 88.9789576532516 | 89.04761904761905 | 840 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 84.16666666666667 | 84.13564733349529 | 84.16666666666667 | 840 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 86.42857142857143 | 86.23948248812134 | 86.42857142857142 | 840 | ok |
| harnet | 4.49 | all | 128 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_right_pocket | 1 | 48.095238095238095 | 38.3401117341003 | 48.095238095238095 | 840 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_right_pocket | 1 | 76.19047619047619 | 76.21300412073344 | 76.1904761904762 | 840 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_right_pocket | 1 | 76.19047619047619 | 76.21300412073344 | 76.1904761904762 | 840 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_right_pocket | 1 | 75.11904761904762 | 74.9701667665893 | 75.11904761904763 | 840 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_right_pocket | 1 | 86.19047619047619 | 86.17008512589564 | 86.19047619047619 | 840 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_right_pocket | 1 | 81.78571428571428 | 81.73441042265979 | 81.78571428571429 | 840 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_right_pocket | 1 | 83.21428571428572 | 83.12349164346696 | 83.21428571428572 | 840 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_right_pocket | 1 | 89.52380952380953 | 89.51370849519222 | 89.52380952380953 | 840 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_right_pocket | 1 | 86.30952380952381 | 86.18859123900923 | 86.3095238095238 | 840 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_right_pocket | 1 | 85.47619047619047 | 85.25033960339898 | 85.47619047619047 | 840 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_right_pocket | 1 | 90.83333333333333 | 90.81514699845296 | 90.83333333333333 | 840 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_right_pocket | 1 | 85.47619047619047 | 85.236467072595 | 85.47619047619047 | 840 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_right_pocket | 1 | 85.59523809523809 | 85.21610293587757 | 85.5952380952381 | 840 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_right_pocket | 1 | 94.76190476190476 | 94.76836697731554 | 94.76190476190479 | 840 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_right_pocket | 1 | 85.23809523809524 | 84.88373813068242 | 85.23809523809524 | 840 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_right_pocket | 1 | 87.61904761904762 | 87.06449340236458 | 87.61904761904763 | 840 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_right_pocket | 1 | 96.07142857142857 | 96.07331662909701 | 96.07142857142857 | 840 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_right_pocket | 1 | 88.09523809523809 | 87.7182254881624 | 88.09523809523809 | 840 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_right_pocket | 1 | 89.16666666666667 | 88.6226271879153 | 89.16666666666667 | 840 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_right_pocket | 1 | 97.61904761904762 | 97.61806339643255 | 97.61904761904762 | 840 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_right_pocket | 1 | 87.85714285714286 | 87.38111347256017 | 87.85714285714285 | 840 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_right_pocket | 1 | 88.57142857142857 | 88.13085374862749 | 88.57142857142858 | 840 | ok |
| unimts | 68.61 | all | 128 | True | False | phone_right_pocket | 1 |  |  |  |  | n/a |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_left_pocket | 1 | 47.73809523809524 | 37.85103627977345 | 47.73809523809524 | 840 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_left_pocket | 1 | 77.14285714285715 | 77.09702181553249 | 77.14285714285715 | 840 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_left_pocket | 1 | 77.14285714285715 | 77.09702181553249 | 77.14285714285715 | 840 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_left_pocket | 1 | 76.54761904761904 | 76.52154730657557 | 76.54761904761907 | 840 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_left_pocket | 1 | 84.88095238095238 | 84.87581984682728 | 84.88095238095238 | 840 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_left_pocket | 1 | 83.80952380952381 | 83.73130021850412 | 83.80952380952381 | 840 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_left_pocket | 1 | 83.57142857142857 | 83.46308501527089 | 83.57142857142857 | 840 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_left_pocket | 1 | 89.16666666666667 | 89.14537149990505 | 89.16666666666664 | 840 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_left_pocket | 1 | 85.23809523809524 | 85.06706426154142 | 85.23809523809524 | 840 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_left_pocket | 1 | 85.11904761904762 | 84.94113565006712 | 85.11904761904762 | 840 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_left_pocket | 1 | 90.95238095238095 | 90.97026286898787 | 90.95238095238095 | 840 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_left_pocket | 1 | 88.57142857142857 | 88.19660259947985 | 88.57142857142856 | 840 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_left_pocket | 1 | 87.85714285714286 | 87.4926586247606 | 87.85714285714286 | 840 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_left_pocket | 1 | 92.97619047619048 | 92.96478385914145 | 92.97619047619047 | 840 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_left_pocket | 1 | 87.38095238095238 | 87.1607401339378 | 87.38095238095237 | 840 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_left_pocket | 1 | 89.16666666666667 | 88.87961726165949 | 89.16666666666666 | 840 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_left_pocket | 1 | 94.28571428571428 | 94.2973427549889 | 94.28571428571428 | 840 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_left_pocket | 1 | 87.97619047619048 | 87.68305662836153 | 87.97619047619048 | 840 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_left_pocket | 1 | 89.64285714285715 | 89.32040085645954 | 89.64285714285715 | 840 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_left_pocket | 1 | 96.78571428571429 | 96.77904543336201 | 96.78571428571429 | 840 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_left_pocket | 1 | 88.80952380952381 | 88.3733471766614 | 88.80952380952381 | 840 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_left_pocket | 1 | 90.83333333333333 | 90.44496165677342 | 90.83333333333334 | 840 | ok |
| unimts | 68.61 | all | 128 | True | False | phone_left_pocket | 1 |  |  |  |  | n/a |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_belt | 1 | 51.42857142857142 | 41.62688255312704 | 51.42857142857144 | 840 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_belt | 1 | 71.30952380952381 | 71.20268857135682 | 71.30952380952381 | 840 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_belt | 1 | 71.30952380952381 | 71.20268857135682 | 71.30952380952381 | 840 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_belt | 1 | 69.28571428571428 | 69.15882251940228 | 69.28571428571429 | 840 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_belt | 1 | 77.14285714285715 | 77.10688691410049 | 77.14285714285714 | 840 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_belt | 1 | 74.28571428571429 | 74.35053614579971 | 74.28571428571429 | 840 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_belt | 1 | 73.33333333333333 | 73.22706643832491 | 73.33333333333334 | 840 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_belt | 1 | 80.83333333333333 | 80.8111547214319 | 80.83333333333333 | 840 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_belt | 1 | 79.4047619047619 | 79.44485543293031 | 79.40476190476191 | 840 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_belt | 1 | 78.45238095238095 | 78.31331572577362 | 78.45238095238095 | 840 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_belt | 1 | 86.54761904761905 | 86.4912618011279 | 86.54761904761902 | 840 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_belt | 1 | 80.11904761904762 | 80.08242232005834 | 80.11904761904762 | 840 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_belt | 1 | 80.83333333333333 | 80.72084972930033 | 80.83333333333333 | 840 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_belt | 1 | 88.69047619047619 | 88.6826862200154 | 88.6904761904762 | 840 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_belt | 1 | 80.23809523809524 | 80.2994390892174 | 80.23809523809524 | 840 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_belt | 1 | 82.85714285714286 | 82.73396498919018 | 82.85714285714285 | 840 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_belt | 1 | 89.64285714285715 | 89.65179675441955 | 89.64285714285714 | 840 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_belt | 1 | 81.30952380952381 | 81.36888020889234 | 81.30952380952382 | 840 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_belt | 1 | 87.02380952380952 | 86.96918208259798 | 87.02380952380952 | 840 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_belt | 1 | 92.61904761904762 | 92.62197742830044 | 92.61904761904763 | 840 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_belt | 1 | 82.97619047619048 | 82.86214873376207 | 82.97619047619048 | 840 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_belt | 1 | 90.83333333333333 | 90.79303733154089 | 90.83333333333333 | 840 | ok |
| unimts | 68.61 | all | 128 | True | False | phone_belt | 1 |  |  |  |  | n/a |
| unimts | 68.61 | native_zero_support | 0 | True | False | watch_wrist_proxy | 1 | 44.285714285714285 | 35.12526784262646 | 44.28571428571429 | 840 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | watch_wrist_proxy | 1 | 66.78571428571428 | 67.10428590461436 | 66.78571428571429 | 840 | ok |
| unimts | 68.61 | prototype | 1 | True | False | watch_wrist_proxy | 1 | 66.78571428571428 | 67.10428590461436 | 66.78571428571429 | 840 | ok |
| unimts | 68.61 | ridge | 1 | True | False | watch_wrist_proxy | 1 | 64.04761904761904 | 63.807568791383154 | 64.04761904761905 | 840 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | watch_wrist_proxy | 1 | 72.85714285714285 | 72.77641524036777 | 72.85714285714286 | 840 | ok |
| unimts | 68.61 | prototype | 2 | True | False | watch_wrist_proxy | 1 | 69.4047619047619 | 69.52129354368411 | 69.4047619047619 | 840 | ok |
| unimts | 68.61 | ridge | 2 | True | False | watch_wrist_proxy | 1 | 67.38095238095238 | 67.21080962144431 | 67.38095238095238 | 840 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | watch_wrist_proxy | 1 | 77.38095238095238 | 77.51352258944699 | 77.38095238095238 | 840 | ok |
| unimts | 68.61 | prototype | 4 | True | False | watch_wrist_proxy | 1 | 71.30952380952381 | 71.52552821248514 | 71.3095238095238 | 840 | ok |
| unimts | 68.61 | ridge | 4 | True | False | watch_wrist_proxy | 1 | 70.35714285714286 | 70.37030865786403 | 70.35714285714285 | 840 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | watch_wrist_proxy | 1 | 86.42857142857143 | 86.49579994334816 | 86.42857142857142 | 840 | ok |
| unimts | 68.61 | prototype | 8 | True | False | watch_wrist_proxy | 1 | 73.45238095238096 | 73.66469364599531 | 73.45238095238093 | 840 | ok |
| unimts | 68.61 | ridge | 8 | True | False | watch_wrist_proxy | 1 | 73.92857142857143 | 73.88389303713676 | 73.92857142857144 | 840 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | watch_wrist_proxy | 1 | 84.76190476190476 | 84.84408472541787 | 84.76190476190476 | 840 | ok |
| unimts | 68.61 | prototype | 16 | True | False | watch_wrist_proxy | 1 | 75.83333333333333 | 75.92997625214119 | 75.83333333333334 | 840 | ok |
| unimts | 68.61 | ridge | 16 | True | False | watch_wrist_proxy | 1 | 77.5 | 77.41792922152665 | 77.5 | 840 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | watch_wrist_proxy | 1 | 88.45238095238095 | 88.5112377377709 | 88.45238095238095 | 840 | ok |
| unimts | 68.61 | prototype | 32 | True | False | watch_wrist_proxy | 1 | 76.30952380952381 | 76.328943153389 | 76.30952380952381 | 840 | ok |
| unimts | 68.61 | ridge | 32 | True | False | watch_wrist_proxy | 1 | 79.76190476190477 | 79.71174965003992 | 79.76190476190477 | 840 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | watch_wrist_proxy | 1 | 88.21428571428571 | 88.32676412254821 | 88.21428571428571 | 840 | ok |
| unimts | 68.61 | prototype | 64 | True | False | watch_wrist_proxy | 1 | 77.73809523809524 | 77.84465192716425 | 77.73809523809524 | 840 | ok |
| unimts | 68.61 | ridge | 64 | True | False | watch_wrist_proxy | 1 | 83.69047619047619 | 83.64921983586521 | 83.69047619047619 | 840 | ok |
| unimts | 68.61 | all | 128 | True | False | watch_wrist_proxy | 1 |  |  |  |  | n/a |
| unimts | 68.61 | zero_support | 0 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| unimts | 68.61 | 1nn | 1 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 75.47619047619048 | 75.58581121574522 | 75.47619047619048 | 840 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 75.47619047619048 | 75.58581121574522 | 75.47619047619048 | 840 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 74.76190476190476 | 74.67333535666037 | 74.76190476190474 | 840 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 82.85714285714286 | 82.9033660203377 | 82.85714285714285 | 840 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 79.4047619047619 | 79.48219938637762 | 79.40476190476191 | 840 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 79.4047619047619 | 79.38095440394768 | 79.40476190476191 | 840 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 88.09523809523809 | 88.12651231186898 | 88.0952380952381 | 840 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 81.66666666666667 | 81.65646649575862 | 81.66666666666667 | 840 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 82.14285714285714 | 82.11138883776073 | 82.14285714285714 | 840 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 91.54761904761905 | 91.56456651389175 | 91.54761904761905 | 840 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 84.76190476190476 | 84.73696400291459 | 84.76190476190476 | 840 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 87.5 | 87.46187886757292 | 87.49999999999999 | 840 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 93.45238095238095 | 93.47075242268357 | 93.45238095238095 | 840 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 85.11904761904762 | 85.06811187198596 | 85.11904761904762 | 840 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 88.21428571428571 | 88.15947770678275 | 88.21428571428571 | 840 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 96.07142857142857 | 96.08003860575775 | 96.07142857142857 | 840 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 85.47619047619047 | 85.26511716377841 | 85.47619047619047 | 840 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 89.88095238095238 | 89.78586497750808 | 89.88095238095238 | 840 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 96.30952380952381 | 96.3049363513354 | 96.30952380952381 | 840 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 85.23809523809524 | 85.09499384549135 | 85.23809523809524 | 840 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 91.07142857142857 | 91.01671600227544 | 91.07142857142856 | 840 | ok |
| unimts | 68.61 | all | 128 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_right_pocket | 1 | 14.285714285714285 | 3.571428571428571 | 14.285714285714285 | 840 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_right_pocket | 1 | 22.61904761904762 | 22.945876951891602 | 22.61904761904762 | 840 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_right_pocket | 1 | 22.61904761904762 | 22.945876951891602 | 22.61904761904762 | 840 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_right_pocket | 1 | 22.976190476190474 | 22.7701790791281 | 22.976190476190478 | 840 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_right_pocket | 1 | 25.0 | 25.251021717616357 | 24.999999999999996 | 840 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_right_pocket | 1 | 25.357142857142854 | 25.64105298981824 | 25.35714285714286 | 840 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_right_pocket | 1 | 25.0 | 24.678980259480614 | 25.0 | 840 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_right_pocket | 1 | 33.33333333333333 | 33.83093511159566 | 33.33333333333334 | 840 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_right_pocket | 1 | 31.547619047619047 | 31.58215499406466 | 31.54761904761904 | 840 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_right_pocket | 1 | 26.785714285714285 | 25.3247176880174 | 26.785714285714292 | 840 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_right_pocket | 1 | 38.21428571428571 | 39.0188904567564 | 38.21428571428571 | 840 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_right_pocket | 1 | 29.04761904761905 | 28.699741260459753 | 29.04761904761904 | 840 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_right_pocket | 1 | 28.333333333333332 | 26.036686712978387 | 28.333333333333332 | 840 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_right_pocket | 1 | 46.07142857142857 | 46.83961829592711 | 46.07142857142858 | 840 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_right_pocket | 1 | 32.261904761904766 | 31.491820676544553 | 32.261904761904766 | 840 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_right_pocket | 1 | 31.30952380952381 | 28.062696354521737 | 31.309523809523814 | 840 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_right_pocket | 1 | 47.023809523809526 | 47.643121650769366 | 47.023809523809526 | 840 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_right_pocket | 1 | 33.92857142857143 | 32.88457615175649 | 33.92857142857143 | 840 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_right_pocket | 1 | 35.23809523809524 | 31.31958703607417 | 35.23809523809524 | 840 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_right_pocket | 1 | 54.761904761904766 | 55.27136078752448 | 54.76190476190476 | 840 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_right_pocket | 1 | 34.404761904761905 | 32.74465350633821 | 34.4047619047619 | 840 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_right_pocket | 1 | 39.88095238095239 | 35.60625912738953 | 39.88095238095239 | 840 | ok |
| normwear | 1293.86 | all | 128 | True | False | phone_right_pocket | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_left_pocket | 1 | 14.285714285714285 | 3.571428571428571 | 14.285714285714285 | 840 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_left_pocket | 1 | 25.595238095238095 | 25.923974222713515 | 25.5952380952381 | 840 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_left_pocket | 1 | 25.595238095238095 | 25.923974222713515 | 25.5952380952381 | 840 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_left_pocket | 1 | 26.666666666666668 | 25.980622620643366 | 26.666666666666668 | 840 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_left_pocket | 1 | 28.095238095238095 | 28.710098669387474 | 28.095238095238095 | 840 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_left_pocket | 1 | 28.333333333333332 | 28.515824679023037 | 28.333333333333332 | 840 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_left_pocket | 1 | 25.71428571428571 | 24.705934390884348 | 25.71428571428571 | 840 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_left_pocket | 1 | 34.404761904761905 | 34.75103704078833 | 34.40476190476191 | 840 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_left_pocket | 1 | 31.785714285714285 | 31.655308918756557 | 31.78571428571429 | 840 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_left_pocket | 1 | 29.404761904761905 | 27.353416995022812 | 29.4047619047619 | 840 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_left_pocket | 1 | 40.595238095238095 | 41.143363855368506 | 40.595238095238095 | 840 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_left_pocket | 1 | 35.23809523809524 | 34.619642315030795 | 35.238095238095234 | 840 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_left_pocket | 1 | 30.714285714285715 | 26.104016442276183 | 30.71428571428571 | 840 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_left_pocket | 1 | 45.95238095238095 | 46.606233369343755 | 45.952380952380956 | 840 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_left_pocket | 1 | 34.76190476190476 | 33.339584292413186 | 34.76190476190476 | 840 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_left_pocket | 1 | 32.73809523809524 | 27.93189527910776 | 32.73809523809524 | 840 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_left_pocket | 1 | 48.45238095238095 | 49.24304838022472 | 48.45238095238095 | 840 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_left_pocket | 1 | 37.38095238095238 | 35.52780952994642 | 37.38095238095239 | 840 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_left_pocket | 1 | 35.595238095238095 | 29.52043498350677 | 35.5952380952381 | 840 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_left_pocket | 1 | 55.47619047619048 | 56.026230098064936 | 55.47619047619048 | 840 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_left_pocket | 1 | 36.547619047619044 | 34.0852436222796 | 36.547619047619044 | 840 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_left_pocket | 1 | 38.452380952380956 | 32.6402172693186 | 38.45238095238095 | 840 | ok |
| normwear | 1293.86 | all | 128 | True | False | phone_left_pocket | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_belt | 1 | 14.285714285714285 | 3.571428571428571 | 14.285714285714285 | 840 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_belt | 1 | 28.214285714285715 | 27.8568424917803 | 28.214285714285715 | 840 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_belt | 1 | 28.214285714285715 | 27.8568424917803 | 28.214285714285715 | 840 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_belt | 1 | 28.214285714285715 | 27.704126392928487 | 28.21428571428572 | 840 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_belt | 1 | 28.92857142857143 | 28.763093810797898 | 28.928571428571427 | 840 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_belt | 1 | 29.28571428571429 | 29.174925809853846 | 29.28571428571428 | 840 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_belt | 1 | 27.380952380952383 | 25.599439621026075 | 27.380952380952383 | 840 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_belt | 1 | 33.69047619047619 | 33.6048704057751 | 33.69047619047619 | 840 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_belt | 1 | 30.952380952380953 | 30.928430799614315 | 30.95238095238095 | 840 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_belt | 1 | 27.142857142857142 | 24.690583089896126 | 27.142857142857142 | 840 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_belt | 1 | 36.666666666666664 | 36.72027416883246 | 36.66666666666667 | 840 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_belt | 1 | 33.214285714285715 | 33.17431679003278 | 33.214285714285715 | 840 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_belt | 1 | 28.214285714285715 | 25.513135313936264 | 28.214285714285715 | 840 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_belt | 1 | 44.88095238095238 | 44.870747091175396 | 44.88095238095238 | 840 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_belt | 1 | 34.166666666666664 | 34.25723239956165 | 34.166666666666664 | 840 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_belt | 1 | 30.952380952380953 | 28.834016644583798 | 30.95238095238096 | 840 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_belt | 1 | 49.166666666666664 | 49.23828844053777 | 49.166666666666664 | 840 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_belt | 1 | 34.04761904761905 | 34.220429551586946 | 34.04761904761905 | 840 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_belt | 1 | 34.04761904761905 | 31.659477859188144 | 34.04761904761905 | 840 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_belt | 1 | 51.42857142857142 | 51.585681573238794 | 51.42857142857144 | 840 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_belt | 1 | 33.33333333333333 | 33.54655073169967 | 33.333333333333336 | 840 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_belt | 1 | 37.142857142857146 | 35.04962189638974 | 37.142857142857146 | 840 | ok |
| normwear | 1293.86 | all | 128 | True | False | phone_belt | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | watch_wrist_proxy | 1 | 14.285714285714285 | 3.571428571428571 | 14.285714285714285 | 840 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | watch_wrist_proxy | 1 | 29.88095238095238 | 29.038181359116678 | 29.88095238095238 | 840 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | watch_wrist_proxy | 1 | 29.88095238095238 | 29.038181359116678 | 29.88095238095238 | 840 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | watch_wrist_proxy | 1 | 29.642857142857142 | 28.01605237892342 | 29.642857142857142 | 840 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | watch_wrist_proxy | 1 | 37.5 | 36.66435635169212 | 37.50000000000001 | 840 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | watch_wrist_proxy | 1 | 37.976190476190474 | 37.16342847612704 | 37.976190476190474 | 840 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | watch_wrist_proxy | 1 | 33.214285714285715 | 29.293221607736392 | 33.214285714285715 | 840 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | watch_wrist_proxy | 1 | 42.023809523809526 | 41.77016234067412 | 42.02380952380952 | 840 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | watch_wrist_proxy | 1 | 41.19047619047619 | 39.51972181122878 | 41.19047619047618 | 840 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | watch_wrist_proxy | 1 | 36.30952380952381 | 28.756540395913216 | 36.30952380952381 | 840 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | watch_wrist_proxy | 1 | 46.785714285714285 | 46.86006234465531 | 46.78571428571429 | 840 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | watch_wrist_proxy | 1 | 41.30952380952381 | 39.07600681401871 | 41.30952380952381 | 840 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | watch_wrist_proxy | 1 | 37.26190476190476 | 27.226850250893946 | 37.26190476190476 | 840 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | watch_wrist_proxy | 1 | 50.0 | 49.886888823704965 | 50.0 | 840 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | watch_wrist_proxy | 1 | 44.285714285714285 | 42.20553256021669 | 44.28571428571428 | 840 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | watch_wrist_proxy | 1 | 39.04761904761905 | 27.199724629220384 | 39.04761904761905 | 840 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | watch_wrist_proxy | 1 | 56.19047619047619 | 56.03796033147196 | 56.19047619047619 | 840 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | watch_wrist_proxy | 1 | 44.761904761904766 | 42.61174612239269 | 44.761904761904766 | 840 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | watch_wrist_proxy | 1 | 39.76190476190476 | 27.924383289624146 | 39.76190476190476 | 840 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | watch_wrist_proxy | 1 | 61.78571428571429 | 61.65656256942383 | 61.78571428571429 | 840 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | watch_wrist_proxy | 1 | 43.69047619047619 | 41.375250649953166 | 43.6904761904762 | 840 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | watch_wrist_proxy | 1 | 44.285714285714285 | 34.825223661059745 | 44.28571428571428 | 840 | ok |
| normwear | 1293.86 | all | 128 | True | False | watch_wrist_proxy | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | zero_support | 0 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 27.26190476190476 | 27.3247632851363 | 27.261904761904766 | 840 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 27.26190476190476 | 27.3247632851363 | 27.261904761904766 | 840 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 27.380952380952383 | 26.835423269483755 | 27.38095238095238 | 840 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 36.19047619047619 | 36.35197459824784 | 36.1904761904762 | 840 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 34.04761904761905 | 34.25135133280985 | 34.04761904761905 | 840 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 28.69047619047619 | 27.964960118752273 | 28.69047619047619 | 840 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 44.047619047619044 | 44.13008265144915 | 44.04761904761906 | 840 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 37.142857142857146 | 37.22945897721067 | 37.142857142857146 | 840 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 33.57142857142857 | 31.305033748246082 | 33.57142857142858 | 840 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 50.83333333333333 | 50.952544121972196 | 50.83333333333333 | 840 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 40.595238095238095 | 40.43892985983014 | 40.595238095238095 | 840 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 32.73809523809524 | 30.038891317091398 | 32.73809523809524 | 840 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 57.97619047619048 | 58.183427017349274 | 57.97619047619048 | 840 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 41.66666666666667 | 41.378908320477066 | 41.666666666666664 | 840 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 35.476190476190474 | 31.928546795966433 | 35.476190476190474 | 840 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 62.142857142857146 | 62.012696255266434 | 62.14285714285713 | 840 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 46.30952380952381 | 45.95315375866103 | 46.30952380952381 | 840 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 40.0 | 37.115638800966764 | 40.0 | 840 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 63.92857142857142 | 64.15332183652123 | 63.92857142857144 | 840 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 46.42857142857143 | 46.56354671549478 | 46.42857142857143 | 840 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 44.285714285714285 | 42.575924341468365 | 44.28571428571429 | 840 | ok |
| normwear | 1293.86 | all | 128 | True | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_right_pocket | 1 | 42.5 | 35.5666981999991 | 42.5 | 840 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_right_pocket | 1 | 70.47619047619048 | 70.4686018733912 | 70.47619047619047 | 840 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_right_pocket | 1 | 70.47619047619048 | 70.4686018733912 | 70.47619047619047 | 840 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_right_pocket | 1 | 65.35714285714286 | 63.14045010937933 | 65.35714285714286 | 840 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_right_pocket | 1 | 81.78571428571428 | 81.83546770506894 | 81.78571428571428 | 840 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_right_pocket | 1 | 79.28571428571428 | 79.27743768878986 | 79.28571428571428 | 840 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_right_pocket | 1 | 69.4047619047619 | 67.06003276311768 | 69.4047619047619 | 840 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_right_pocket | 1 | 88.33333333333333 | 88.4241320510107 | 88.33333333333334 | 840 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_right_pocket | 1 | 82.5 | 82.73916901014154 | 82.5 | 840 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_right_pocket | 1 | 73.09523809523809 | 71.11134665137399 | 73.09523809523809 | 840 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_right_pocket | 1 | 91.54761904761905 | 91.53121070365935 | 91.54761904761905 | 840 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_right_pocket | 1 | 89.52380952380953 | 89.53605004057074 | 89.5238095238095 | 840 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_right_pocket | 1 | 76.54761904761904 | 74.53131297046967 | 76.54761904761905 | 840 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_right_pocket | 1 | 93.69047619047619 | 93.68185783039866 | 93.69047619047619 | 840 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_right_pocket | 1 | 91.78571428571428 | 91.83562214972373 | 91.78571428571428 | 840 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_right_pocket | 1 | 81.54761904761905 | 79.95138583508408 | 81.54761904761905 | 840 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_right_pocket | 1 | 93.45238095238095 | 93.45715045876537 | 93.45238095238095 | 840 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_right_pocket | 1 | 93.45238095238095 | 93.50590873795522 | 93.45238095238095 | 840 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_right_pocket | 1 | 85.35714285714285 | 84.36641494137643 | 85.35714285714285 | 840 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_right_pocket | 1 | 93.45238095238095 | 93.4210856495002 | 93.45238095238095 | 840 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_right_pocket | 1 | 94.04761904761905 | 94.09940952166106 | 94.04761904761905 | 840 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_right_pocket | 1 | 87.97619047619048 | 87.41437120363119 | 87.97619047619048 | 840 | ok |
| limubert_x | 0.055 | all | 128 | False | False | phone_right_pocket | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_left_pocket | 1 | 38.095238095238095 | 29.92618280604475 | 38.0952380952381 | 840 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_left_pocket | 1 | 76.54761904761904 | 76.92570280762945 | 76.54761904761905 | 840 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_left_pocket | 1 | 76.54761904761904 | 76.92570280762945 | 76.54761904761905 | 840 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_left_pocket | 1 | 66.30952380952381 | 63.80931889306739 | 66.30952380952381 | 840 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_left_pocket | 1 | 87.14285714285714 | 87.21654296333516 | 87.14285714285714 | 840 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_left_pocket | 1 | 82.85714285714286 | 83.02204131494265 | 82.85714285714285 | 840 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_left_pocket | 1 | 66.07142857142857 | 62.45451385908877 | 66.07142857142857 | 840 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_left_pocket | 1 | 89.28571428571429 | 89.33135187900413 | 89.28571428571429 | 840 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_left_pocket | 1 | 87.5 | 87.56664573676811 | 87.50000000000001 | 840 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_left_pocket | 1 | 69.16666666666667 | 65.89236653092847 | 69.16666666666667 | 840 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_left_pocket | 1 | 90.11904761904762 | 90.18365232129196 | 90.11904761904762 | 840 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_left_pocket | 1 | 89.04761904761904 | 89.10926431147975 | 89.04761904761905 | 840 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_left_pocket | 1 | 76.07142857142857 | 74.14127789488079 | 76.07142857142858 | 840 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_left_pocket | 1 | 90.5952380952381 | 90.72795958229747 | 90.5952380952381 | 840 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_left_pocket | 1 | 91.07142857142857 | 91.08834252788405 | 91.07142857142857 | 840 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_left_pocket | 1 | 83.57142857142857 | 82.74700839330742 | 83.57142857142857 | 840 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_left_pocket | 1 | 91.42857142857143 | 91.57859663049767 | 91.42857142857143 | 840 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_left_pocket | 1 | 92.14285714285714 | 92.15904439018897 | 92.14285714285715 | 840 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_left_pocket | 1 | 87.85714285714286 | 87.26139149664067 | 87.85714285714286 | 840 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_left_pocket | 1 | 91.9047619047619 | 91.97944670482195 | 91.90476190476191 | 840 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_left_pocket | 1 | 91.78571428571428 | 91.8017022810262 | 91.78571428571428 | 840 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_left_pocket | 1 | 89.28571428571429 | 88.98749595460046 | 89.2857142857143 | 840 | ok |
| limubert_x | 0.055 | all | 128 | False | False | phone_left_pocket | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_belt | 1 | 36.904761904761905 | 31.916779428099456 | 36.9047619047619 | 840 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_belt | 1 | 62.26190476190476 | 62.7324449278829 | 62.26190476190476 | 840 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_belt | 1 | 62.26190476190476 | 62.7324449278829 | 62.26190476190476 | 840 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_belt | 1 | 54.52380952380952 | 51.51927859126878 | 54.52380952380953 | 840 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_belt | 1 | 75.71428571428571 | 75.82441956392667 | 75.71428571428571 | 840 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_belt | 1 | 62.61904761904762 | 63.06102913435259 | 62.619047619047606 | 840 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_belt | 1 | 56.07142857142857 | 53.19174401454446 | 56.07142857142857 | 840 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_belt | 1 | 77.26190476190476 | 77.45923336410472 | 77.26190476190477 | 840 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_belt | 1 | 59.523809523809526 | 60.56277998184096 | 59.523809523809526 | 840 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_belt | 1 | 56.547619047619044 | 52.555693444137056 | 56.547619047619044 | 840 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_belt | 1 | 78.45238095238095 | 78.74903977062105 | 78.45238095238095 | 840 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_belt | 1 | 65.5952380952381 | 66.08794696504529 | 65.5952380952381 | 840 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_belt | 1 | 63.095238095238095 | 58.07923841221937 | 63.0952380952381 | 840 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_belt | 1 | 79.64285714285714 | 79.90070004472442 | 79.64285714285715 | 840 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_belt | 1 | 68.21428571428572 | 68.55888448252037 | 68.21428571428571 | 840 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_belt | 1 | 65.35714285714286 | 56.956435344453816 | 65.35714285714286 | 840 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_belt | 1 | 80.23809523809524 | 80.51524832052786 | 80.23809523809523 | 840 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_belt | 1 | 73.80952380952381 | 74.0890389178924 | 73.8095238095238 | 840 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_belt | 1 | 65.95238095238095 | 57.743663559464295 | 65.95238095238096 | 840 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_belt | 1 | 79.76190476190477 | 80.03044687472298 | 79.76190476190476 | 840 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_belt | 1 | 77.5 | 77.26190281177419 | 77.5 | 840 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_belt | 1 | 65.83333333333333 | 57.11691676086773 | 65.83333333333333 | 840 | ok |
| limubert_x | 0.055 | all | 128 | False | False | phone_belt | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | watch_wrist_proxy | 1 | 22.023809523809522 | 17.766501094855588 | 22.023809523809522 | 840 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | watch_wrist_proxy | 1 | 68.92857142857143 | 68.97576790754304 | 68.92857142857143 | 840 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | watch_wrist_proxy | 1 | 68.92857142857143 | 68.97576790754304 | 68.92857142857143 | 840 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | watch_wrist_proxy | 1 | 62.976190476190474 | 60.71524457620894 | 62.976190476190474 | 840 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | watch_wrist_proxy | 1 | 77.14285714285715 | 77.16988102878328 | 77.14285714285715 | 840 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | watch_wrist_proxy | 1 | 74.04761904761905 | 73.99223193699888 | 74.04761904761905 | 840 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | watch_wrist_proxy | 1 | 66.54761904761905 | 64.65088099964122 | 66.54761904761905 | 840 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | watch_wrist_proxy | 1 | 83.21428571428572 | 83.10862574141956 | 83.21428571428571 | 840 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | watch_wrist_proxy | 1 | 77.73809523809524 | 77.47160737705815 | 77.73809523809526 | 840 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | watch_wrist_proxy | 1 | 72.38095238095238 | 70.86969577773955 | 72.38095238095238 | 840 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | watch_wrist_proxy | 1 | 84.52380952380952 | 84.54422681540407 | 84.52380952380952 | 840 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | watch_wrist_proxy | 1 | 78.80952380952381 | 78.65717584836204 | 78.80952380952381 | 840 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | watch_wrist_proxy | 1 | 75.83333333333333 | 74.67522576694921 | 75.83333333333331 | 840 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | watch_wrist_proxy | 1 | 88.21428571428571 | 88.29691431463631 | 88.21428571428572 | 840 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | watch_wrist_proxy | 1 | 79.16666666666666 | 78.96267934089977 | 79.16666666666666 | 840 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | watch_wrist_proxy | 1 | 83.45238095238095 | 82.96790557580866 | 83.45238095238095 | 840 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | watch_wrist_proxy | 1 | 90.83333333333333 | 90.8574500666954 | 90.83333333333331 | 840 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | watch_wrist_proxy | 1 | 82.26190476190476 | 82.13951929331567 | 82.26190476190477 | 840 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | watch_wrist_proxy | 1 | 85.59523809523809 | 85.38426898265207 | 85.5952380952381 | 840 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | watch_wrist_proxy | 1 | 90.71428571428571 | 90.71860973086243 | 90.71428571428572 | 840 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | watch_wrist_proxy | 1 | 82.02380952380952 | 81.90461116405878 | 82.02380952380953 | 840 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | watch_wrist_proxy | 1 | 87.61904761904762 | 87.40249114177877 | 87.61904761904763 | 840 | ok |
| limubert_x | 0.055 | all | 128 | False | False | watch_wrist_proxy | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | zero_support | 0 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 84.16666666666667 | 84.50105987831871 | 84.16666666666669 | 840 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 84.16666666666667 | 84.50105987831871 | 84.16666666666669 | 840 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 66.66666666666666 | 62.092338175385734 | 66.66666666666667 | 840 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 91.30952380952381 | 91.39301247304135 | 91.30952380952382 | 840 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 89.52380952380953 | 89.6546670008369 | 89.52380952380953 | 840 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 66.9047619047619 | 61.658294204324115 | 66.90476190476191 | 840 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 90.95238095238095 | 91.06326105403417 | 90.95238095238095 | 840 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 90.5952380952381 | 90.7456605554014 | 90.5952380952381 | 840 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 69.28571428571428 | 63.432814961468566 | 69.28571428571428 | 840 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 92.26190476190477 | 92.28571934987374 | 92.26190476190476 | 840 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 92.26190476190477 | 92.33536647300072 | 92.26190476190477 | 840 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 72.02380952380952 | 65.64488200866393 | 72.02380952380952 | 840 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 92.14285714285714 | 92.19611889388428 | 92.14285714285714 | 840 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 92.5 | 92.50191645550603 | 92.5 | 840 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 74.16666666666667 | 67.97349551156744 | 74.16666666666666 | 840 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 92.85714285714286 | 92.8827655168704 | 92.85714285714286 | 840 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 93.80952380952381 | 93.8008428905426 | 93.80952380952381 | 840 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 77.97619047619048 | 74.04302463222213 | 77.97619047619048 | 840 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 92.5 | 92.55459319610362 | 92.5 | 840 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 94.04761904761905 | 94.06776000196218 | 94.04761904761905 | 840 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 | 83.92857142857143 | 82.40795073788424 | 83.92857142857143 | 840 | ok |
| limubert_x | 0.055 | all | 128 | False | False | phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 4 |  |  |  |  | n/a |

### usc_had
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | stream | n_devices | accuracy | f1_macro | balanced_accuracy | n_queries | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halo | 2.203 | halo-classifier | 0 | True | True | phone_hip | 1 | 39.009990917347864 | 35.777213319363995 | 44.66671326466012 | 2202 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | phone_hip | 1 | 36.51226158038147 | 34.37549182496506 | 41.24606235270547 | 2202 | ok |
| halo | 0.789 | 1nn | 1 | True | True | phone_hip | 1 | 53.723887375113534 | 56.02295689414961 | 55.997733278894756 | 2202 | ok |
| halo | 0.789 | prototype | 1 | True | True | phone_hip | 1 | 53.723887375113534 | 56.02295689414961 | 55.997733278894756 | 2202 | ok |
| halo | 0.789 | ridge | 1 | True | True | phone_hip | 1 | 54.40508628519528 | 56.16018659955236 | 56.84143730567456 | 2202 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | phone_hip | 1 | 52.270663033605814 | 53.628978228791965 | 57.127363032488596 | 2202 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | phone_hip | 1 | 54.35967302452316 | 56.5766942211429 | 56.52471662431626 | 2202 | ok |
| halo | 0.789 | 1nn | 2 | True | True | phone_hip | 1 | 58.81017257039055 | 61.1884228229837 | 61.32596070715856 | 2202 | ok |
| halo | 0.789 | prototype | 2 | True | True | phone_hip | 1 | 60.12715712988192 | 62.29504749153866 | 62.503130767186 | 2202 | ok |
| halo | 0.789 | ridge | 2 | True | True | phone_hip | 1 | 60.53587647593097 | 62.132702889574475 | 62.9071993019427 | 2202 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | phone_hip | 1 | 55.449591280653955 | 56.15829019341441 | 60.31908490596464 | 2202 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | phone_hip | 1 | 59.90009082652135 | 62.11030342354397 | 62.151152961998726 | 2202 | ok |
| halo | 0.789 | 1nn | 4 | True | True | phone_hip | 1 | 63.35149863760218 | 65.38614404624967 | 65.46963111446334 | 2202 | ok |
| halo | 0.789 | prototype | 4 | True | True | phone_hip | 1 | 62.48864668483197 | 64.53561697296254 | 64.55933576935035 | 2202 | ok |
| halo | 0.789 | ridge | 4 | True | True | phone_hip | 1 | 61.35331516802906 | 63.038293405039134 | 63.8048686044797 | 2202 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | phone_hip | 1 | 57.72025431425977 | 58.39788593198677 | 62.58963995433828 | 2202 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | phone_hip | 1 | 64.12352406902816 | 65.84319654494142 | 65.79073780976726 | 2202 | ok |
| halo | 0.789 | 1nn | 8 | True | True | phone_hip | 1 | 67.93823796548592 | 69.24489863665353 | 69.43649379451125 | 2202 | ok |
| halo | 0.789 | prototype | 8 | True | True | phone_hip | 1 | 63.89645776566758 | 65.40624123151918 | 65.82699024879042 | 2202 | ok |
| halo | 0.789 | ridge | 8 | True | True | phone_hip | 1 | 65.21344232515894 | 66.17725852340834 | 67.13735119496548 | 2202 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | phone_hip | 1 | 59.945504087193456 | 60.06910755430604 | 64.62762181527505 | 2202 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | phone_hip | 1 | 68.34695731153498 | 69.57025967960291 | 69.61721271098061 | 2202 | ok |
| halo | 0.789 | 1nn | 16 | True | True | phone_hip | 1 | 73.88737511353315 | 74.63225366816486 | 74.56571834319156 | 2202 | ok |
| halo | 0.789 | prototype | 16 | True | True | phone_hip | 1 | 67.3478655767484 | 68.26067440830502 | 68.67143722700412 | 2202 | ok |
| halo | 0.789 | ridge | 16 | True | True | phone_hip | 1 | 69.3006357856494 | 69.67668057691073 | 70.48104216605141 | 2202 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | phone_hip | 1 | 64.03269754768392 | 63.52898709580289 | 67.87909905401571 | 2202 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | phone_hip | 1 | 72.0708446866485 | 73.15456018494439 | 73.03826678149345 | 2202 | ok |
| halo | 0.789 | 1nn | 32 | True | True | phone_hip | 1 | 78.65576748410535 | 78.42277620083186 | 78.55220151633583 | 2202 | ok |
| halo | 0.789 | prototype | 32 | True | True | phone_hip | 1 | 66.57584014532243 | 67.2615305393017 | 67.72783692251525 | 2202 | ok |
| halo | 0.789 | ridge | 32 | True | True | phone_hip | 1 | 71.43505903723887 | 71.55001757841391 | 72.16489514986768 | 2202 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | phone_hip | 1 | 64.94096276112626 | 64.15459716178201 | 68.65741280850362 | 2202 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | phone_hip | 1 | 75.47683923705722 | 75.79094790511132 | 75.7699490063855 | 2202 | ok |
| halo | 0.789 | 1nn | 64 | True | True | phone_hip | 1 | 83.83287920072662 | 83.24282610394583 | 83.37909687887105 | 2202 | ok |
| halo | 0.789 | prototype | 64 | True | True | phone_hip | 1 | 68.30154405086284 | 68.93038474415894 | 69.49204414442688 | 2202 | ok |
| halo | 0.789 | ridge | 64 | True | True | phone_hip | 1 | 75.47683923705722 | 75.3061216383593 | 75.89644896656885 | 2202 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | phone_hip | 1 | 65.57674841053588 | 64.03474922341451 | 69.10069360296173 | 2202 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | phone_hip | 1 | 77.20254314259763 | 77.28533915704827 | 77.26386198754375 | 2202 | ok |
| halo | 0.789 | all | 128 | True | True | phone_hip | 1 |  |  |  |  | n/a |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | phone_hip | 1 | 28.973660308810175 | 28.311933324024263 | 32.211385350888655 | 2202 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | phone_hip | 1 | 33.42415985467756 | 34.53550482870242 | 35.30248800218822 | 2202 | ok |
| harnet | 4.49 | prototype | 1 | False | False | phone_hip | 1 | 33.42415985467756 | 34.53550482870242 | 35.30248800218822 | 2202 | ok |
| harnet | 4.49 | ridge | 1 | False | False | phone_hip | 1 | 34.9227974568574 | 35.695649888017925 | 36.43258497641802 | 2202 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | phone_hip | 1 | 40.463215258855584 | 41.179227491354666 | 41.833457366226796 | 2202 | ok |
| harnet | 4.49 | prototype | 2 | False | False | phone_hip | 1 | 38.69209809264305 | 39.77330636951643 | 40.19210927488658 | 2202 | ok |
| harnet | 4.49 | ridge | 2 | False | False | phone_hip | 1 | 39.782016348773844 | 40.23245345629062 | 41.180323323157175 | 2202 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | phone_hip | 1 | 44.550408719346045 | 45.75808419867452 | 46.22670116058599 | 2202 | ok |
| harnet | 4.49 | prototype | 4 | False | False | phone_hip | 1 | 41.91643960036331 | 42.839962788472704 | 43.21088452695965 | 2202 | ok |
| harnet | 4.49 | ridge | 4 | False | False | phone_hip | 1 | 44.50499545867393 | 44.85365085809417 | 46.020649331992345 | 2202 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | phone_hip | 1 | 51.22615803814714 | 52.37940611905406 | 52.89950624629854 | 2202 | ok |
| harnet | 4.49 | prototype | 8 | False | False | phone_hip | 1 | 45.41326067211626 | 46.41179518448919 | 47.24979927970278 | 2202 | ok |
| harnet | 4.49 | ridge | 8 | False | False | phone_hip | 1 | 48.68301544050863 | 48.81059605846556 | 50.68694733366057 | 2202 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | phone_hip | 1 | 56.176203451407815 | 56.806062795029234 | 57.33476417410376 | 2202 | ok |
| harnet | 4.49 | prototype | 16 | False | False | phone_hip | 1 | 47.09355131698456 | 48.074516146875084 | 48.889259320507406 | 2202 | ok |
| harnet | 4.49 | ridge | 16 | False | False | phone_hip | 1 | 51.72570390554042 | 51.65690594235184 | 53.40590796007475 | 2202 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | phone_hip | 1 | 63.66939146230699 | 64.33776819524876 | 64.73804927225711 | 2202 | ok |
| harnet | 4.49 | prototype | 32 | False | False | phone_hip | 1 | 49.90917347865577 | 50.845108265490225 | 51.781072564972895 | 2202 | ok |
| harnet | 4.49 | ridge | 32 | False | False | phone_hip | 1 | 56.403269754768395 | 56.41816091067543 | 58.417198168742324 | 2202 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | phone_hip | 1 | 68.61943687556766 | 69.59597672005722 | 69.94669912383236 | 2202 | ok |
| harnet | 4.49 | prototype | 64 | False | False | phone_hip | 1 | 51.90735694822888 | 52.423132344629266 | 53.80325409818146 | 2202 | ok |
| harnet | 4.49 | ridge | 64 | False | False | phone_hip | 1 | 60.39963669391463 | 60.41949496561663 | 62.50083346042935 | 2202 | ok |
| harnet | 4.49 | all | 128 | False | False | phone_hip | 1 |  |  |  |  | n/a |
| unimts | 68.61 | native_zero_support | 0 | True | False | phone_hip | 1 | 27.475022706630337 | 25.501010046529878 | 31.330701001738205 | 2202 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | phone_hip | 1 | 36.421435059037236 | 39.22831887609479 | 38.91385062119725 | 2202 | ok |
| unimts | 68.61 | prototype | 1 | True | False | phone_hip | 1 | 36.421435059037236 | 39.22831887609479 | 38.91385062119725 | 2202 | ok |
| unimts | 68.61 | ridge | 1 | True | False | phone_hip | 1 | 37.46594005449591 | 39.077992603991014 | 39.58824275710949 | 2202 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | phone_hip | 1 | 43.32425068119891 | 46.20496253694292 | 45.855002664330584 | 2202 | ok |
| unimts | 68.61 | prototype | 2 | True | False | phone_hip | 1 | 39.782016348773844 | 42.65124710759318 | 42.662376711412776 | 2202 | ok |
| unimts | 68.61 | ridge | 2 | True | False | phone_hip | 1 | 41.008174386920984 | 43.19103291718298 | 43.85649575756614 | 2202 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | phone_hip | 1 | 51.271571298819254 | 53.554734805220164 | 53.20699256070621 | 2202 | ok |
| unimts | 68.61 | prototype | 4 | True | False | phone_hip | 1 | 40.82652134423251 | 44.05009001322973 | 44.126239583379274 | 2202 | ok |
| unimts | 68.61 | ridge | 4 | True | False | phone_hip | 1 | 42.68846503178928 | 45.48708312374334 | 46.17345197496675 | 2202 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | phone_hip | 1 | 55.76748410535877 | 57.42924034942365 | 57.049382002338234 | 2202 | ok |
| unimts | 68.61 | prototype | 8 | True | False | phone_hip | 1 | 43.50590372388737 | 46.69325725098376 | 47.2906901874732 | 2202 | ok |
| unimts | 68.61 | ridge | 8 | True | False | phone_hip | 1 | 48.001816530426886 | 50.522172361784854 | 51.59575675060718 | 2202 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | phone_hip | 1 | 62.125340599455036 | 63.25985526811401 | 63.246519702744855 | 2202 | ok |
| unimts | 68.61 | prototype | 16 | True | False | phone_hip | 1 | 45.27702089009991 | 48.14015390148684 | 48.71941079742228 | 2202 | ok |
| unimts | 68.61 | ridge | 16 | True | False | phone_hip | 1 | 53.17892824704814 | 55.03450353150857 | 55.768120875397145 | 2202 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | phone_hip | 1 | 67.62034514078111 | 68.38264684901608 | 68.52876356760538 | 2202 | ok |
| unimts | 68.61 | prototype | 32 | True | False | phone_hip | 1 | 48.41053587647593 | 50.661728402662845 | 51.69282235023675 | 2202 | ok |
| unimts | 68.61 | ridge | 32 | True | False | phone_hip | 1 | 59.40054495912806 | 59.96401539623232 | 60.86901013123106 | 2202 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | phone_hip | 1 | 72.0708446866485 | 72.45445096852295 | 72.56696471382939 | 2202 | ok |
| unimts | 68.61 | prototype | 64 | True | False | phone_hip | 1 | 49.22797456857403 | 50.87441679801002 | 51.970529712141875 | 2202 | ok |
| unimts | 68.61 | ridge | 64 | True | False | phone_hip | 1 | 62.44323342415985 | 62.65682767413734 | 63.529071374320125 | 2202 | ok |
| unimts | 68.61 | all | 128 | True | False | phone_hip | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | phone_hip | 1 | 9.854677565849228 | 2.2188657480594083 | 8.763837638376383 | 2202 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | phone_hip | 1 | 17.756584922797458 | 17.573957904542148 | 17.963259663068516 | 2202 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | phone_hip | 1 | 17.756584922797458 | 17.573957904542148 | 17.963259663068516 | 2202 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | phone_hip | 1 | 15.89464123524069 | 15.213435978245608 | 15.573820673893044 | 2202 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | phone_hip | 1 | 20.072661217075385 | 19.88559204112008 | 20.168800738327143 | 2202 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | phone_hip | 1 | 18.93732970027248 | 18.47295927863668 | 18.720279747392578 | 2202 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | phone_hip | 1 | 16.03088101725704 | 14.970079406054928 | 15.766532356517459 | 2202 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | phone_hip | 1 | 20.572207084468666 | 20.06872698601091 | 20.033676466678205 | 2202 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | phone_hip | 1 | 20.29972752043597 | 19.761411225128146 | 20.010294786556347 | 2202 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | phone_hip | 1 | 20.118074477747502 | 18.585084968903065 | 19.774957336270614 | 2202 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | phone_hip | 1 | 25.61307901907357 | 25.469705447096096 | 25.521130764206045 | 2202 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | phone_hip | 1 | 21.798365122615802 | 20.984634167995626 | 21.258205999329025 | 2202 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | phone_hip | 1 | 19.891008174386922 | 17.154769187117946 | 18.52995260814676 | 2202 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | phone_hip | 1 | 29.8819255222525 | 29.72853892855789 | 29.793358110160973 | 2202 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | phone_hip | 1 | 22.116257947320616 | 21.013803009569475 | 21.56368180567616 | 2202 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | phone_hip | 1 | 25.295186194368757 | 21.099714537836824 | 23.19491980768406 | 2202 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | phone_hip | 1 | 34.05994550408719 | 33.779613190301696 | 33.96333799572251 | 2202 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | phone_hip | 1 | 23.115349682107176 | 21.90678315309659 | 23.01235314918997 | 2202 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | phone_hip | 1 | 26.067211625794734 | 20.644664180987604 | 23.546664725849265 | 2202 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | phone_hip | 1 | 34.332425068119896 | 34.165470185867626 | 34.345899590544256 | 2202 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | phone_hip | 1 | 23.978201634877383 | 21.778852120416094 | 23.25551937025285 | 2202 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | phone_hip | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | all | 128 | True | False | phone_hip | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | phone_hip | 1 | 13.35149863760218 | 12.810951739204715 | 15.140933146408301 | 2202 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | phone_hip | 1 | 47.275204359673026 | 45.57451989825427 | 45.23849748707392 | 2202 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | phone_hip | 1 | 47.275204359673026 | 45.57451989825427 | 45.23849748707392 | 2202 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | phone_hip | 1 | 43.097184377838325 | 40.67925198048699 | 42.90968413074093 | 2202 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | phone_hip | 1 | 56.176203451407815 | 53.78340364189745 | 53.21473877341265 | 2202 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | phone_hip | 1 | 53.04268846503179 | 50.628781406427535 | 50.089240749015616 | 2202 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | phone_hip | 1 | 49.4550408719346 | 45.74125667088241 | 48.58376522057362 | 2202 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | phone_hip | 1 | 62.17075386012716 | 59.731522116668 | 59.18770809770749 | 2202 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | phone_hip | 1 | 59.218891916439595 | 56.27616332250119 | 55.95427970786011 | 2202 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | phone_hip | 1 | 55.449591280653955 | 51.21116493055518 | 53.70473585535647 | 2202 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | phone_hip | 1 | 69.66394187102634 | 67.49264510849295 | 66.94315469250124 | 2202 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | phone_hip | 1 | 61.21707538601272 | 58.00495664744201 | 57.50255604393754 | 2202 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | phone_hip | 1 | 57.5386012715713 | 52.41328959494436 | 55.63635053049166 | 2202 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | phone_hip | 1 | 75.9763851044505 | 73.81827197767812 | 73.32505366989929 | 2202 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | phone_hip | 1 | 64.89554950045414 | 61.97770856311951 | 61.515210662269936 | 2202 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | phone_hip | 1 | 59.763851044505 | 54.715458141889016 | 57.97515239654457 | 2202 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | phone_hip | 1 | 80.6993642143506 | 78.76358722592435 | 78.27201717153999 | 2202 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | phone_hip | 1 | 64.71389645776566 | 62.0167820999181 | 61.59749284214823 | 2202 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | phone_hip | 1 | 63.53315168029064 | 59.46802175626639 | 61.79239941588769 | 2202 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | phone_hip | 1 | 85.28610354223434 | 83.63246013395039 | 83.36014331067733 | 2202 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | phone_hip | 1 | 67.12079927338783 | 64.47047259337319 | 64.10257788272492 | 2202 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | phone_hip | 1 | 66.62125340599455 | 63.128030440927375 | 65.16785833497082 | 2202 | ok |
| limubert_x | 0.055 | all | 128 | False | False | phone_hip | 1 |  |  |  |  | n/a |

### ut_complex
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | stream | n_devices | accuracy | f1_macro | balanced_accuracy | n_queries | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halo | 2.203 | halo-classifier | 0 | True | True | watch_wrist | 1 | 45.19230769230769 | 39.94905033616438 | 45.19230769230769 | 1560 | ok |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | watch_wrist | 1 | 28.012820512820515 | 25.793328362450303 | 28.012820512820515 | 1560 | ok |
| halo | 0.789 | 1nn | 1 | True | True | watch_wrist | 1 | 71.47435897435898 | 71.35906466571025 | 71.47435897435896 | 1560 | ok |
| halo | 0.789 | prototype | 1 | True | True | watch_wrist | 1 | 71.47435897435898 | 71.35906466571025 | 71.47435897435896 | 1560 | ok |
| halo | 0.789 | ridge | 1 | True | True | watch_wrist | 1 | 73.07692307692307 | 72.79519201533843 | 73.07692307692307 | 1560 | ok |
| halo | 2.203 | halo-classifier | 1 | True | True | watch_wrist | 1 | 75.12820512820512 | 74.82985729629075 | 75.12820512820511 | 1560 | ok |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | watch_wrist | 1 | 72.37179487179488 | 72.21887468715596 | 72.37179487179488 | 1560 | ok |
| halo | 0.789 | 1nn | 2 | True | True | watch_wrist | 1 | 75.0 | 74.91045945857839 | 75.0 | 1560 | ok |
| halo | 0.789 | prototype | 2 | True | True | watch_wrist | 1 | 76.7948717948718 | 76.70785500675534 | 76.7948717948718 | 1560 | ok |
| halo | 0.789 | ridge | 2 | True | True | watch_wrist | 1 | 77.6923076923077 | 77.42195697572272 | 77.69230769230766 | 1560 | ok |
| halo | 2.203 | halo-classifier | 2 | True | True | watch_wrist | 1 | 76.15384615384615 | 75.95507335712216 | 76.15384615384617 | 1560 | ok |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | watch_wrist | 1 | 75.44871794871794 | 75.3082198127612 | 75.44871794871794 | 1560 | ok |
| halo | 0.789 | 1nn | 4 | True | True | watch_wrist | 1 | 81.28205128205128 | 81.1885800827353 | 81.28205128205128 | 1560 | ok |
| halo | 0.789 | prototype | 4 | True | True | watch_wrist | 1 | 82.5 | 82.44182375339892 | 82.5 | 1560 | ok |
| halo | 0.789 | ridge | 4 | True | True | watch_wrist | 1 | 83.91025641025641 | 83.70851173150588 | 83.91025641025642 | 1560 | ok |
| halo | 2.203 | halo-classifier | 4 | True | True | watch_wrist | 1 | 81.08974358974359 | 80.79319830120541 | 81.0897435897436 | 1560 | ok |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | watch_wrist | 1 | 81.92307692307692 | 81.80021993328525 | 81.9230769230769 | 1560 | ok |
| halo | 0.789 | 1nn | 8 | True | True | watch_wrist | 1 | 82.82051282051283 | 82.82247198131816 | 82.82051282051283 | 1560 | ok |
| halo | 0.789 | prototype | 8 | True | True | watch_wrist | 1 | 84.67948717948718 | 84.68758135307053 | 84.67948717948718 | 1560 | ok |
| halo | 0.789 | ridge | 8 | True | True | watch_wrist | 1 | 85.8974358974359 | 85.75551033604565 | 85.8974358974359 | 1560 | ok |
| halo | 2.203 | halo-classifier | 8 | True | True | watch_wrist | 1 | 82.05128205128204 | 81.87428615758766 | 82.05128205128204 | 1560 | ok |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | watch_wrist | 1 | 85.12820512820512 | 85.09131801049746 | 85.12820512820512 | 1560 | ok |
| halo | 0.789 | 1nn | 16 | True | True | watch_wrist | 1 | 84.74358974358974 | 84.73274014970235 | 84.74358974358974 | 1560 | ok |
| halo | 0.789 | prototype | 16 | True | True | watch_wrist | 1 | 85.12820512820512 | 85.13643324205138 | 85.12820512820512 | 1560 | ok |
| halo | 0.789 | ridge | 16 | True | True | watch_wrist | 1 | 87.75641025641025 | 87.62373785161833 | 87.75641025641025 | 1560 | ok |
| halo | 2.203 | halo-classifier | 16 | True | True | watch_wrist | 1 | 83.01282051282051 | 82.94073251419259 | 83.01282051282051 | 1560 | ok |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | watch_wrist | 1 | 86.66666666666667 | 86.58706329793205 | 86.66666666666667 | 1560 | ok |
| halo | 0.789 | 1nn | 32 | True | True | watch_wrist | 1 | 86.6025641025641 | 86.56233866551648 | 86.60256410256412 | 1560 | ok |
| halo | 0.789 | prototype | 32 | True | True | watch_wrist | 1 | 85.7051282051282 | 85.6682769668181 | 85.7051282051282 | 1560 | ok |
| halo | 0.789 | ridge | 32 | True | True | watch_wrist | 1 | 88.33333333333333 | 88.20520120046378 | 88.33333333333333 | 1560 | ok |
| halo | 2.203 | halo-classifier | 32 | True | True | watch_wrist | 1 | 84.35897435897436 | 84.18510904009938 | 84.35897435897436 | 1560 | ok |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | watch_wrist | 1 | 88.14102564102564 | 88.08779934604219 | 88.14102564102565 | 1560 | ok |
| halo | 0.789 | 1nn | 64 | True | True | watch_wrist | 1 | 87.11538461538461 | 87.0254605300906 | 87.1153846153846 | 1560 | ok |
| halo | 0.789 | prototype | 64 | True | True | watch_wrist | 1 | 86.73076923076923 | 86.73683092266421 | 86.73076923076923 | 1560 | ok |
| halo | 0.789 | ridge | 64 | True | True | watch_wrist | 1 | 89.42307692307693 | 89.34673393721155 | 89.4230769230769 | 1560 | ok |
| halo | 2.203 | halo-classifier | 64 | True | True | watch_wrist | 1 | 84.1025641025641 | 83.96214163951673 | 84.1025641025641 | 1560 | ok |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | watch_wrist | 1 | 89.1025641025641 | 89.06115506278016 | 89.1025641025641 | 1560 | ok |
| halo | 0.789 | all | 128 | True | True | watch_wrist | 1 |  |  |  |  | n/a |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | watch_wrist | 1 | 40.12820512820513 | 37.164049424992356 | 40.12820512820513 | 1560 | ok |
| harnet | 4.49 | 1nn | 1 | False | False | watch_wrist | 1 | 55.705128205128204 | 55.23504863961176 | 55.705128205128204 | 1560 | ok |
| harnet | 4.49 | prototype | 1 | False | False | watch_wrist | 1 | 55.705128205128204 | 55.23504863961176 | 55.705128205128204 | 1560 | ok |
| harnet | 4.49 | ridge | 1 | False | False | watch_wrist | 1 | 55.51282051282052 | 54.716823526896185 | 55.51282051282052 | 1560 | ok |
| harnet | 4.49 | 1nn | 2 | False | False | watch_wrist | 1 | 60.83333333333333 | 60.560318546779236 | 60.83333333333333 | 1560 | ok |
| harnet | 4.49 | prototype | 2 | False | False | watch_wrist | 1 | 61.08974358974359 | 60.71320285162234 | 61.08974358974359 | 1560 | ok |
| harnet | 4.49 | ridge | 2 | False | False | watch_wrist | 1 | 60.44871794871794 | 59.61043595703269 | 60.448717948717956 | 1560 | ok |
| harnet | 4.49 | 1nn | 4 | False | False | watch_wrist | 1 | 67.94871794871796 | 67.79198576692734 | 67.94871794871794 | 1560 | ok |
| harnet | 4.49 | prototype | 4 | False | False | watch_wrist | 1 | 65.83333333333333 | 65.66822235800555 | 65.83333333333333 | 1560 | ok |
| harnet | 4.49 | ridge | 4 | False | False | watch_wrist | 1 | 67.56410256410257 | 66.95662549688186 | 67.56410256410257 | 1560 | ok |
| harnet | 4.49 | 1nn | 8 | False | False | watch_wrist | 1 | 70.12820512820512 | 70.14318381065878 | 70.12820512820514 | 1560 | ok |
| harnet | 4.49 | prototype | 8 | False | False | watch_wrist | 1 | 69.61538461538461 | 69.4573944401912 | 69.61538461538463 | 1560 | ok |
| harnet | 4.49 | ridge | 8 | False | False | watch_wrist | 1 | 74.61538461538461 | 73.58937501684323 | 74.6153846153846 | 1560 | ok |
| harnet | 4.49 | 1nn | 16 | False | False | watch_wrist | 1 | 73.3974358974359 | 73.33791162534911 | 73.3974358974359 | 1560 | ok |
| harnet | 4.49 | prototype | 16 | False | False | watch_wrist | 1 | 71.08974358974359 | 71.0567624545708 | 71.0897435897436 | 1560 | ok |
| harnet | 4.49 | ridge | 16 | False | False | watch_wrist | 1 | 79.2948717948718 | 78.61512603210753 | 79.2948717948718 | 1560 | ok |
| harnet | 4.49 | 1nn | 32 | False | False | watch_wrist | 1 | 76.66666666666667 | 76.72629010039952 | 76.66666666666667 | 1560 | ok |
| harnet | 4.49 | prototype | 32 | False | False | watch_wrist | 1 | 72.05128205128204 | 72.09219863086206 | 72.05128205128204 | 1560 | ok |
| harnet | 4.49 | ridge | 32 | False | False | watch_wrist | 1 | 81.53846153846153 | 80.63145167972785 | 81.53846153846153 | 1560 | ok |
| harnet | 4.49 | 1nn | 64 | False | False | watch_wrist | 1 | 78.52564102564102 | 78.64700619000786 | 78.52564102564104 | 1560 | ok |
| harnet | 4.49 | prototype | 64 | False | False | watch_wrist | 1 | 73.3974358974359 | 73.44604609186011 | 73.3974358974359 | 1560 | ok |
| harnet | 4.49 | ridge | 64 | False | False | watch_wrist | 1 | 83.91025641025641 | 83.25993802490751 | 83.9102564102564 | 1560 | ok |
| harnet | 4.49 | all | 128 | False | False | watch_wrist | 1 |  |  |  |  | n/a |
| unimts | 68.61 | native_zero_support | 0 | True | False | watch_wrist | 1 | 26.02564102564103 | 18.80910080168021 | 26.02564102564103 | 1560 | ok |
| unimts | 68.61 | 1nn | 1 | True | False | watch_wrist | 1 | 56.92307692307692 | 57.07154677812976 | 56.92307692307692 | 1560 | ok |
| unimts | 68.61 | prototype | 1 | True | False | watch_wrist | 1 | 56.92307692307692 | 57.07154677812976 | 56.92307692307692 | 1560 | ok |
| unimts | 68.61 | ridge | 1 | True | False | watch_wrist | 1 | 54.807692307692314 | 54.47179698881578 | 54.807692307692314 | 1560 | ok |
| unimts | 68.61 | 1nn | 2 | True | False | watch_wrist | 1 | 62.5 | 62.50055643522673 | 62.500000000000014 | 1560 | ok |
| unimts | 68.61 | prototype | 2 | True | False | watch_wrist | 1 | 60.38461538461538 | 60.502546977848205 | 60.38461538461538 | 1560 | ok |
| unimts | 68.61 | ridge | 2 | True | False | watch_wrist | 1 | 58.84615384615385 | 58.58218446898154 | 58.84615384615385 | 1560 | ok |
| unimts | 68.61 | 1nn | 4 | True | False | watch_wrist | 1 | 67.56410256410257 | 67.83438300565477 | 67.56410256410257 | 1560 | ok |
| unimts | 68.61 | prototype | 4 | True | False | watch_wrist | 1 | 65.0 | 65.32679337310762 | 64.99999999999999 | 1560 | ok |
| unimts | 68.61 | ridge | 4 | True | False | watch_wrist | 1 | 62.756410256410255 | 62.3740141413643 | 62.756410256410255 | 1560 | ok |
| unimts | 68.61 | 1nn | 8 | True | False | watch_wrist | 1 | 71.66666666666667 | 71.77601653013687 | 71.66666666666667 | 1560 | ok |
| unimts | 68.61 | prototype | 8 | True | False | watch_wrist | 1 | 67.88461538461539 | 68.09467309546154 | 67.88461538461537 | 1560 | ok |
| unimts | 68.61 | ridge | 8 | True | False | watch_wrist | 1 | 67.05128205128204 | 66.29322654974867 | 67.05128205128206 | 1560 | ok |
| unimts | 68.61 | 1nn | 16 | True | False | watch_wrist | 1 | 74.23076923076923 | 74.44407344836912 | 74.23076923076923 | 1560 | ok |
| unimts | 68.61 | prototype | 16 | True | False | watch_wrist | 1 | 69.48717948717949 | 69.77179172155932 | 69.48717948717947 | 1560 | ok |
| unimts | 68.61 | ridge | 16 | True | False | watch_wrist | 1 | 71.85897435897436 | 71.26277977851636 | 71.85897435897436 | 1560 | ok |
| unimts | 68.61 | 1nn | 32 | True | False | watch_wrist | 1 | 76.15384615384615 | 76.32301639701952 | 76.15384615384616 | 1560 | ok |
| unimts | 68.61 | prototype | 32 | True | False | watch_wrist | 1 | 69.35897435897435 | 69.65218452838481 | 69.35897435897435 | 1560 | ok |
| unimts | 68.61 | ridge | 32 | True | False | watch_wrist | 1 | 73.58974358974359 | 73.1799559030833 | 73.58974358974358 | 1560 | ok |
| unimts | 68.61 | 1nn | 64 | True | False | watch_wrist | 1 | 78.65384615384615 | 78.86791684453036 | 78.65384615384615 | 1560 | ok |
| unimts | 68.61 | prototype | 64 | True | False | watch_wrist | 1 | 71.85897435897436 | 72.03466367551954 | 71.85897435897436 | 1560 | ok |
| unimts | 68.61 | ridge | 64 | True | False | watch_wrist | 1 | 78.78205128205128 | 78.57437841444548 | 78.78205128205128 | 1560 | ok |
| unimts | 68.61 | all | 128 | True | False | watch_wrist | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | native_zero_support | 0 | True | False | watch_wrist | 1 | 7.6923076923076925 | 1.0989010989010988 | 7.6923076923076925 | 1560 | ok |
| normwear | 1293.86 | 1nn | 1 | True | False | watch_wrist | 1 | 26.602564102564102 | 26.39845773057678 | 26.60256410256411 | 1560 | ok |
| normwear | 1293.86 | prototype | 1 | True | False | watch_wrist | 1 | 26.602564102564102 | 26.39845773057678 | 26.60256410256411 | 1560 | ok |
| normwear | 1293.86 | ridge | 1 | True | False | watch_wrist | 1 | 20.0 | 18.057567597203253 | 20.000000000000004 | 1560 | ok |
| normwear | 1293.86 | 1nn | 2 | True | False | watch_wrist | 1 | 28.78205128205128 | 28.60238035577869 | 28.78205128205128 | 1560 | ok |
| normwear | 1293.86 | prototype | 2 | True | False | watch_wrist | 1 | 29.615384615384617 | 29.14970837831178 | 29.61538461538461 | 1560 | ok |
| normwear | 1293.86 | ridge | 2 | True | False | watch_wrist | 1 | 20.76923076923077 | 17.346486680733808 | 20.769230769230766 | 1560 | ok |
| normwear | 1293.86 | 1nn | 4 | True | False | watch_wrist | 1 | 33.01282051282051 | 32.97320360526828 | 33.01282051282052 | 1560 | ok |
| normwear | 1293.86 | prototype | 4 | True | False | watch_wrist | 1 | 31.538461538461537 | 30.583566258977307 | 31.538461538461537 | 1560 | ok |
| normwear | 1293.86 | ridge | 4 | True | False | watch_wrist | 1 | 22.94871794871795 | 17.69555247390112 | 22.948717948717945 | 1560 | ok |
| normwear | 1293.86 | 1nn | 8 | True | False | watch_wrist | 1 | 37.88461538461539 | 37.94425253263839 | 37.88461538461539 | 1560 | ok |
| normwear | 1293.86 | prototype | 8 | True | False | watch_wrist | 1 | 34.23076923076923 | 33.12275498096659 | 34.230769230769226 | 1560 | ok |
| normwear | 1293.86 | ridge | 8 | True | False | watch_wrist | 1 | 25.833333333333336 | 18.801162283514845 | 25.83333333333333 | 1560 | ok |
| normwear | 1293.86 | 1nn | 16 | True | False | watch_wrist | 1 | 40.96153846153846 | 40.79600010844236 | 40.96153846153846 | 1560 | ok |
| normwear | 1293.86 | prototype | 16 | True | False | watch_wrist | 1 | 34.03846153846154 | 32.226846480050334 | 34.03846153846154 | 1560 | ok |
| normwear | 1293.86 | ridge | 16 | True | False | watch_wrist | 1 | 28.653846153846153 | 20.43236992111211 | 28.653846153846153 | 1560 | ok |
| normwear | 1293.86 | 1nn | 32 | True | False | watch_wrist | 1 | 46.73076923076923 | 46.91173652156889 | 46.730769230769226 | 1560 | ok |
| normwear | 1293.86 | prototype | 32 | True | False | watch_wrist | 1 | 35.64102564102564 | 33.822363839690354 | 35.64102564102564 | 1560 | ok |
| normwear | 1293.86 | ridge | 32 | True | False | watch_wrist | 1 | 33.84615384615385 | 24.276309833620598 | 33.84615384615385 | 1560 | ok |
| normwear | 1293.86 | 1nn | 64 | True | False | watch_wrist | 1 | 51.66666666666667 | 51.8534678844917 | 51.66666666666667 | 1560 | ok |
| normwear | 1293.86 | prototype | 64 | True | False | watch_wrist | 1 | 35.64102564102564 | 33.88670299309805 | 35.641025641025635 | 1560 | ok |
| normwear | 1293.86 | ridge | 64 | True | False | watch_wrist | 1 |  |  |  |  | n/a |
| normwear | 1293.86 | all | 128 | True | False | watch_wrist | 1 |  |  |  |  | n/a |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | watch_wrist | 1 | 10.128205128205128 | 9.120708925968477 | 10.128205128205128 | 1560 | ok |
| limubert_x | 0.055 | 1nn | 1 | False | False | watch_wrist | 1 | 59.87179487179487 | 59.33750326441943 | 59.87179487179487 | 1560 | ok |
| limubert_x | 0.055 | prototype | 1 | False | False | watch_wrist | 1 | 59.87179487179487 | 59.33750326441943 | 59.87179487179487 | 1560 | ok |
| limubert_x | 0.055 | ridge | 1 | False | False | watch_wrist | 1 | 47.179487179487175 | 42.7531739817156 | 47.179487179487175 | 1560 | ok |
| limubert_x | 0.055 | 1nn | 2 | False | False | watch_wrist | 1 | 61.47435897435898 | 61.0188665826965 | 61.474358974358964 | 1560 | ok |
| limubert_x | 0.055 | prototype | 2 | False | False | watch_wrist | 1 | 59.87179487179487 | 58.71476223495456 | 59.87179487179488 | 1560 | ok |
| limubert_x | 0.055 | ridge | 2 | False | False | watch_wrist | 1 | 48.58974358974359 | 43.08262779158592 | 48.58974358974359 | 1560 | ok |
| limubert_x | 0.055 | 1nn | 4 | False | False | watch_wrist | 1 | 70.96153846153847 | 71.01651822467953 | 70.96153846153844 | 1560 | ok |
| limubert_x | 0.055 | prototype | 4 | False | False | watch_wrist | 1 | 64.74358974358975 | 63.71103072103866 | 64.74358974358974 | 1560 | ok |
| limubert_x | 0.055 | ridge | 4 | False | False | watch_wrist | 1 | 50.57692307692307 | 43.99032222536902 | 50.57692307692307 | 1560 | ok |
| limubert_x | 0.055 | 1nn | 8 | False | False | watch_wrist | 1 | 76.6025641025641 | 76.35327165102359 | 76.6025641025641 | 1560 | ok |
| limubert_x | 0.055 | prototype | 8 | False | False | watch_wrist | 1 | 67.62820512820514 | 66.44464079685119 | 67.62820512820512 | 1560 | ok |
| limubert_x | 0.055 | ridge | 8 | False | False | watch_wrist | 1 | 55.38461538461539 | 48.812353782557416 | 55.38461538461538 | 1560 | ok |
| limubert_x | 0.055 | 1nn | 16 | False | False | watch_wrist | 1 | 78.97435897435898 | 78.89208365430729 | 78.97435897435898 | 1560 | ok |
| limubert_x | 0.055 | prototype | 16 | False | False | watch_wrist | 1 | 70.7051282051282 | 69.54082451130051 | 70.7051282051282 | 1560 | ok |
| limubert_x | 0.055 | ridge | 16 | False | False | watch_wrist | 1 | 59.03846153846154 | 52.71546821760802 | 59.03846153846154 | 1560 | ok |
| limubert_x | 0.055 | 1nn | 32 | False | False | watch_wrist | 1 | 83.65384615384616 | 83.51643002520588 | 83.65384615384615 | 1560 | ok |
| limubert_x | 0.055 | prototype | 32 | False | False | watch_wrist | 1 | 70.44871794871796 | 69.04363322961296 | 70.44871794871794 | 1560 | ok |
| limubert_x | 0.055 | ridge | 32 | False | False | watch_wrist | 1 | 63.141025641025635 | 57.50939186261898 | 63.14102564102565 | 1560 | ok |
| limubert_x | 0.055 | 1nn | 64 | False | False | watch_wrist | 1 | 85.12820512820512 | 85.07090807712795 | 85.12820512820511 | 1560 | ok |
| limubert_x | 0.055 | prototype | 64 | False | False | watch_wrist | 1 | 71.34615384615385 | 69.90822781650633 | 71.34615384615385 | 1560 | ok |
| limubert_x | 0.055 | ridge | 64 | False | False | watch_wrist | 1 | 68.01282051282051 | 63.8568794912173 | 68.01282051282051 | 1560 | ok |
| limubert_x | 0.055 | all | 128 | False | False | watch_wrist | 1 |  |  |  |  | n/a |

### Dataset-Balanced Mean
| model | parameters_m | readout | k | native_open_set_labels | native_few_shot_adaptation | accuracy | f1_macro | n_datasets |
|---|---|---|---|---|---|---|---|---|
| halo | 0.789 | 1nn | 1 | True | True | 62.772 | 62.259 | 6 |
| halo | 0.789 | 1nn | 2 | True | True | 67.053 | 66.762 | 6 |
| halo | 0.789 | 1nn | 4 | True | True | 70.582 | 70.358 | 6 |
| halo | 0.789 | 1nn | 8 | True | True | 72.569 | 72.463 | 6 |
| halo | 0.789 | 1nn | 16 | True | True | 74.493 | 74.379 | 6 |
| halo | 0.789 | 1nn | 32 | True | True | 75.967 | 75.256 | 6 |
| halo | 0.789 | 1nn | 64 | True | True | 79.006 | 78.884 | 6 |
| halo | 0.789 | 1nn | 128 | True | True | 95.15 | 93.823 | 1 |
| halo | 2.203 | halo-classifier | 0 | True | True | 54.105 | 50.147 | 6 |
| halo | 2.203 | halo-classifier | 1 | True | True | 64.891 | 63.903 | 6 |
| halo | 2.203 | halo-classifier | 2 | True | True | 67.557 | 66.763 | 6 |
| halo | 2.203 | halo-classifier | 4 | True | True | 69.538 | 68.704 | 6 |
| halo | 2.203 | halo-classifier | 8 | True | True | 70.903 | 70.044 | 6 |
| halo | 2.203 | halo-classifier | 16 | True | True | 72.496 | 71.202 | 6 |
| halo | 2.203 | halo-classifier | 32 | True | True | 72.844 | 71.512 | 6 |
| halo | 2.203 | halo-classifier | 64 | True | True | 74.643 | 73.046 | 6 |
| halo | 2.203 | halo-classifier | 128 | True | True | 90.093 | 87.32 | 1 |
| halo | 2.203 | halo-classifier-residual-off | 1 | True | True | 62.963 | 62.36 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 2 | True | True | 67.676 | 67.285 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 4 | True | True | 71.677 | 71.242 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 8 | True | True | 74.109 | 73.899 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 16 | True | True | 75.796 | 75.616 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 32 | True | True | 77.149 | 76.395 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 64 | True | True | 78.992 | 78.905 | 6 |
| halo | 2.203 | halo-classifier-residual-off | 128 | True | True | 93.86 | 92.231 | 1 |
| halo | 0.789 | prototype | 1 | True | True | 62.772 | 62.259 | 6 |
| halo | 0.789 | prototype | 2 | True | True | 67.793 | 67.231 | 6 |
| halo | 0.789 | prototype | 4 | True | True | 71.002 | 70.534 | 6 |
| halo | 0.789 | prototype | 8 | True | True | 72.735 | 72.253 | 6 |
| halo | 0.789 | prototype | 16 | True | True | 75.048 | 74.34 | 6 |
| halo | 0.789 | prototype | 32 | True | True | 75.018 | 74.501 | 6 |
| halo | 0.789 | prototype | 64 | True | True | 77.179 | 76.77 | 6 |
| halo | 0.789 | prototype | 128 | True | True | 91.022 | 89.18 | 1 |
| halo | 0.789 | ridge | 1 | True | True | 63.615 | 62.941 | 6 |
| halo | 0.789 | ridge | 2 | True | True | 68.653 | 67.958 | 6 |
| halo | 0.789 | ridge | 4 | True | True | 71.763 | 71.262 | 6 |
| halo | 0.789 | ridge | 8 | True | True | 74.337 | 73.84 | 6 |
| halo | 0.789 | ridge | 16 | True | True | 76.749 | 75.825 | 6 |
| halo | 0.789 | ridge | 32 | True | True | 77.861 | 76.962 | 6 |
| halo | 0.789 | ridge | 64 | True | True | 80.051 | 79.602 | 6 |
| halo | 0.789 | ridge | 128 | True | True | 94.118 | 92.8 | 1 |
| halo | 0.789 | training-bank-1nn-conse | 0 | True | True | 43.388 | 38.486 | 6 |
| harnet | 4.49 | 1nn | 1 | False | False | 48.93 | 48.326 | 6 |
| harnet | 4.49 | 1nn | 2 | False | False | 54.779 | 54.256 | 6 |
| harnet | 4.49 | 1nn | 4 | False | False | 58.544 | 58.531 | 6 |
| harnet | 4.49 | 1nn | 8 | False | False | 60.818 | 60.877 | 6 |
| harnet | 4.49 | 1nn | 16 | False | False | 63.244 | 63.074 | 6 |
| harnet | 4.49 | 1nn | 32 | False | False | 66.136 | 65.983 | 6 |
| harnet | 4.49 | 1nn | 64 | False | False | 68.282 | 68.516 | 6 |
| harnet | 4.49 | 1nn | 128 | False | False | 85.707 | 86.3 | 1 |
| harnet | 4.49 | prototype | 1 | False | False | 48.93 | 48.326 | 6 |
| harnet | 4.49 | prototype | 2 | False | False | 53.597 | 52.992 | 6 |
| harnet | 4.49 | prototype | 4 | False | False | 57.324 | 56.925 | 6 |
| harnet | 4.49 | prototype | 8 | False | False | 59.139 | 59.054 | 6 |
| harnet | 4.49 | prototype | 16 | False | False | 61.316 | 61.017 | 6 |
| harnet | 4.49 | prototype | 32 | False | False | 61.797 | 61.502 | 6 |
| harnet | 4.49 | prototype | 64 | False | False | 63.842 | 63.665 | 6 |
| harnet | 4.49 | prototype | 128 | False | False | 79.721 | 79.604 | 1 |
| harnet | 4.49 | ridge | 1 | False | False | 49.554 | 48.645 | 6 |
| harnet | 4.49 | ridge | 2 | False | False | 54.03 | 52.972 | 6 |
| harnet | 4.49 | ridge | 4 | False | False | 58.3 | 57.616 | 6 |
| harnet | 4.49 | ridge | 8 | False | False | 62.235 | 61.784 | 6 |
| harnet | 4.49 | ridge | 16 | False | False | 65.46 | 64.934 | 6 |
| harnet | 4.49 | ridge | 32 | False | False | 67.507 | 66.919 | 6 |
| harnet | 4.49 | ridge | 64 | False | False | 71.607 | 71.369 | 6 |
| harnet | 4.49 | ridge | 128 | False | False | 87.255 | 87.859 | 1 |
| harnet | 4.49 | training-bank-1nn-conse | 0 | False | False | 40.829 | 36.935 | 6 |
| limubert_x | 0.055 | 1nn | 1 | False | False | 52.359 | 51.623 | 5 |
| limubert_x | 0.055 | 1nn | 2 | False | False | 58.568 | 57.668 | 5 |
| limubert_x | 0.055 | 1nn | 4 | False | False | 64.502 | 63.579 | 5 |
| limubert_x | 0.055 | 1nn | 8 | False | False | 68.325 | 67.415 | 5 |
| limubert_x | 0.055 | 1nn | 16 | False | False | 71.038 | 70.277 | 5 |
| limubert_x | 0.055 | 1nn | 32 | False | False | 74.841 | 74.2 | 5 |
| limubert_x | 0.055 | 1nn | 64 | False | False | 76.152 | 75.636 | 5 |
| limubert_x | 0.055 | 1nn | 128 | False | False | 92.724 | 91.672 | 1 |
| limubert_x | 0.055 | prototype | 1 | False | False | 52.359 | 51.623 | 5 |
| limubert_x | 0.055 | prototype | 2 | False | False | 55.364 | 54.316 | 5 |
| limubert_x | 0.055 | prototype | 4 | False | False | 59.826 | 58.455 | 5 |
| limubert_x | 0.055 | prototype | 8 | False | False | 61.631 | 60.304 | 5 |
| limubert_x | 0.055 | prototype | 16 | False | False | 64.294 | 62.861 | 5 |
| limubert_x | 0.055 | prototype | 32 | False | False | 65.46 | 63.895 | 5 |
| limubert_x | 0.055 | prototype | 64 | False | False | 66.076 | 64.173 | 5 |
| limubert_x | 0.055 | prototype | 128 | False | False | 76.883 | 76.159 | 1 |
| limubert_x | 0.055 | ridge | 1 | False | False | 46.484 | 43.758 | 5 |
| limubert_x | 0.055 | ridge | 2 | False | False | 48.821 | 45.271 | 5 |
| limubert_x | 0.055 | ridge | 4 | False | False | 52.634 | 48.618 | 5 |
| limubert_x | 0.055 | ridge | 8 | False | False | 57.036 | 52.52 | 5 |
| limubert_x | 0.055 | ridge | 16 | False | False | 59.878 | 55.221 | 5 |
| limubert_x | 0.055 | ridge | 32 | False | False | 63.086 | 58.299 | 5 |
| limubert_x | 0.055 | ridge | 64 | False | False | 66.063 | 61.64 | 5 |
| limubert_x | 0.055 | ridge | 128 | False | False | 77.193 | 74.029 | 1 |
| limubert_x | 0.055 | training-bank-1nn-conse | 0 | False | False | 27.718 | 24.276 | 5 |
| normwear | 1293.86 | 1nn | 1 | True | False | 23.758 | 23.215 | 6 |
| normwear | 1293.86 | 1nn | 2 | True | False | 25.802 | 25.208 | 6 |
| normwear | 1293.86 | 1nn | 4 | True | False | 28.323 | 27.741 | 6 |
| normwear | 1293.86 | 1nn | 8 | True | False | 32.127 | 31.592 | 6 |
| normwear | 1293.86 | 1nn | 16 | True | False | 36.129 | 35.559 | 6 |
| normwear | 1293.86 | 1nn | 32 | True | False | 39.029 | 38.702 | 6 |
| normwear | 1293.86 | 1nn | 64 | True | False | 41.8 | 41.694 | 6 |
| normwear | 1293.86 | 1nn | 128 | True | False | 53.922 | 54.345 | 1 |
| normwear | 1293.86 | native_zero_support | 0 | True | False | 14.64 | 3.569 | 6 |
| normwear | 1293.86 | prototype | 1 | True | False | 23.758 | 23.215 | 6 |
| normwear | 1293.86 | prototype | 2 | True | False | 26.272 | 25.51 | 6 |
| normwear | 1293.86 | prototype | 4 | True | False | 27.461 | 26.46 | 6 |
| normwear | 1293.86 | prototype | 8 | True | False | 28.904 | 27.627 | 6 |
| normwear | 1293.86 | prototype | 16 | True | False | 29.342 | 27.689 | 6 |
| normwear | 1293.86 | prototype | 32 | True | False | 29.947 | 28.241 | 6 |
| normwear | 1293.86 | prototype | 64 | True | False | 30.272 | 28.087 | 6 |
| normwear | 1293.86 | prototype | 128 | True | False | 31.631 | 28.735 | 1 |
| normwear | 1293.86 | ridge | 1 | True | False | 22.059 | 20.772 | 6 |
| normwear | 1293.86 | ridge | 2 | True | False | 22.841 | 20.795 | 6 |
| normwear | 1293.86 | ridge | 4 | True | False | 24.671 | 21.617 | 6 |
| normwear | 1293.86 | ridge | 8 | True | False | 25.59 | 21.433 | 6 |
| normwear | 1293.86 | ridge | 16 | True | False | 28.614 | 23.51 | 6 |
| normwear | 1293.86 | ridge | 32 | True | False | 29.89 | 24.15 | 6 |
| normwear | 1293.86 | ridge | 64 | True | False | 31.621 | 26.344 | 4 |
| unimts | 68.61 | 1nn | 1 | True | False | 53.932 | 54.463 | 6 |
| unimts | 68.61 | 1nn | 2 | True | False | 59.886 | 60.783 | 6 |
| unimts | 68.61 | 1nn | 4 | True | False | 64.501 | 65.289 | 6 |
| unimts | 68.61 | 1nn | 8 | True | False | 67.173 | 67.912 | 6 |
| unimts | 68.61 | 1nn | 16 | True | False | 70.096 | 70.232 | 6 |
| unimts | 68.61 | 1nn | 32 | True | False | 71.986 | 72.182 | 6 |
| unimts | 68.61 | 1nn | 64 | True | False | 75.568 | 75.931 | 6 |
| unimts | 68.61 | 1nn | 128 | True | False | 90.351 | 90.592 | 1 |
| unimts | 68.61 | native_zero_support | 0 | True | False | 38.573 | 30.625 | 6 |
| unimts | 68.61 | prototype | 1 | True | False | 53.932 | 54.463 | 6 |
| unimts | 68.61 | prototype | 2 | True | False | 56.895 | 57.812 | 6 |
| unimts | 68.61 | prototype | 4 | True | False | 59.316 | 60.602 | 6 |
| unimts | 68.61 | prototype | 8 | True | False | 61.541 | 62.909 | 6 |
| unimts | 68.61 | prototype | 16 | True | False | 63.064 | 63.949 | 6 |
| unimts | 68.61 | prototype | 32 | True | False | 65.32 | 65.925 | 6 |
| unimts | 68.61 | prototype | 64 | True | False | 67.54 | 68.087 | 6 |
| unimts | 68.61 | prototype | 128 | True | False | 81.889 | 83.443 | 1 |
| unimts | 68.61 | ridge | 1 | True | False | 52.61 | 52.844 | 6 |
| unimts | 68.61 | ridge | 2 | True | False | 56.722 | 57.415 | 6 |
| unimts | 68.61 | ridge | 4 | True | False | 59.829 | 60.881 | 6 |
| unimts | 68.61 | ridge | 8 | True | False | 63.343 | 64.233 | 6 |
| unimts | 68.61 | ridge | 16 | True | False | 67.095 | 67.45 | 6 |
| unimts | 68.61 | ridge | 32 | True | False | 69.676 | 69.597 | 6 |
| unimts | 68.61 | ridge | 64 | True | False | 73.721 | 73.728 | 6 |
| unimts | 68.61 | ridge | 128 | True | False | 87.564 | 88.684 | 1 |
