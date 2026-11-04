# Attribution: war and distracter stems

The WAV files under `war/` and `distracter/` are processed excerpts (mono, at most 10 s, level-normalised,
some denoised; `scripts/build_stems.py`) of clips from FSD50K (Fonseca et al., "FSD50K: an open dataset of
human-labeled sound events", IEEE/ACM TASLP 30, 2022; https://zenodo.org/records/4060432), originally from
Freesound (https://freesound.org). Each clip keeps its own licence, listed below with its uploader.
CC BY clips require this attribution to stay with them. `manifest.csv` holds the full processing record.

`data/processed/war/war_track.wav` (101 MB) is not in git: GitHub refuses files over 100 MB. It is rebuilt
exactly from these stems and a fixed seed by `scripts/war_track.py`; its `events.csv` and `phases.csv` are here.

| File | Class | FSD50K id (split) | Title | Uploader | Licence |
|---|---|---|---|---|---|
| `distracter/aircraft/fsd50k_eval_122807.wav` | aircraft | 122807 (eval) | plane_air_italy_boeing.mp3 | geodylabs | CC BY 3.0 |
| `distracter/aircraft/fsd50k_eval_221665.wav` | aircraft | 221665 (eval) | CommercialJetTakeoffBinaural.WAV | daveincamas | CC BY 3.0 |
| `distracter/aircraft/fsd50k_eval_325845.wav` | aircraft | 325845 (eval) | Airplane inside ambient- cessna 414. condenser .mp3 | Daveshu88 | CC0 1.0 |
| `distracter/birds/fsd50k_eval_417743.wav` | birds | 417743 (eval) | amb-bird-chirping-001.flac | Starscade | CC0 1.0 |
| `distracter/birds/fsd50k_eval_427369.wav` | birds | 427369 (eval) | birds swarm in bush.wav | Douglas711 | CC0 1.0 |
| `distracter/buzzing_insect/fsd50k_dev_233783.wav` | buzzing_insect | 233783 (dev) | BUMBLEBEE 12.wav | delphidebrain | CC BY 3.0 |
| `distracter/buzzing_insect/fsd50k_dev_273214.wav` | buzzing_insect | 273214 (dev) | BeesBuzzing2May2nd2015bitrate48.WAV | kvgarlic | CC0 1.0 |
| `distracter/buzzing_insect/fsd50k_eval_395593.wav` | buzzing_insect | 395593 (eval) | Fly.wav | BeeProductive | CC0 1.0 |
| `distracter/car/fsd50k_eval_336754.wav` | car | 336754 (eval) | Car Drive by 03.WAV | luminadii | CC BY 3.0 |
| `distracter/car/fsd50k_eval_336758.wav` | car | 336758 (eval) | Car drive by 05.WAV | luminadii | CC BY 3.0 |
| `distracter/car/fsd50k_eval_336762.wav` | car | 336762 (eval) | Car drive by 07.WAV | luminadii | CC BY 3.0 |
| `distracter/chainsaw/fsd50k_dev_157611.wav` | chainsaw | 157611 (dev) | Exterior_Chainsaw.wav | DJillcom | CC0 1.0 |
| `distracter/chainsaw/fsd50k_dev_345994.wav` | chainsaw | 345994 (dev) | Chainsaw_2.WAV | ivolipa | CC0 1.0 |
| `distracter/chainsaw/fsd50k_dev_431739.wav` | chainsaw | 431739 (dev) | Chainsaw Cut.wav | Stefan021 | CC0 1.0 |
| `distracter/dog/fsd50k_eval_102547.wav` | dog | 102547 (eval) | dog barking 3 (Rottweiler).aif | Tomlija | CC BY 3.0 |
| `distracter/dog/fsd50k_eval_328055.wav` | dog | 328055 (eval) | BIG DOGS_Continues Barking Labrador Retriever (SFX).wav | GuntherDorksen | CC BY 3.0 |
| `distracter/dog/fsd50k_eval_380440.wav` | dog | 380440 (eval) | Dog Bark + Mysterious Gurgle.wav | FullMetalJedi | CC0 1.0 |
| `distracter/explosion/fsd50k_eval_211436.wav` | explosion | 211436 (eval) | Explosion002.wav | klangfabrik | CC0 1.0 |
| `distracter/explosion/fsd50k_eval_220062.wav` | explosion | 220062 (eval) | Explosion004.wav | klangfabrik | CC0 1.0 |
| `distracter/explosion/fsd50k_eval_95812.wav` | explosion | 95812 (eval) | red_flame.wav | cormi | CC0 1.0 |
| `distracter/generator/fsd50k_dev_164306.wav` | generator | 164306 (dev) | GeneratorEngine.wav | thegreenguerilla | CC BY 3.0 |
| `distracter/generator/fsd50k_dev_367175.wav` | generator | 367175 (dev) | generator small 01 | DiscoveryME | CC0 1.0 |
| `distracter/generator/fsd50k_eval_406110.wav` | generator | 406110 (eval) | Generator.wav | Yin_Yang_Jake007 | CC0 1.0 |
| `distracter/gunfire/fsd50k_eval_174290.wav` | gunfire | 174290 (eval) | REC_1_AT897_STR.wav | ingudios | CC BY 3.0 |
| `distracter/gunfire/fsd50k_eval_204949.wav` | gunfire | 204949 (eval) | m4-carbine improved.wav | tehlordoswag420 | CC BY 3.0 |
| `distracter/gunfire/fsd50k_eval_249146.wav` | gunfire | 249146 (eval) | 50 CAL Machine Gun.wav | klangfabrik | CC0 1.0 |
| `distracter/helicopter/fsd50k_dev_181071.wav` | helicopter | 181071 (dev) | helicopter1.wav | dwareing | CC0 1.0 |
| `distracter/helicopter/fsd50k_eval_263458.wav` | helicopter | 263458 (eval) | Cobra_Helicopter_Flyover_Ext.wav | Cell31_Sound_Productions | CC BY 3.0 |
| `distracter/helicopter/fsd50k_eval_322179.wav` | helicopter | 322179 (eval) | NH90_flyby.aif | ikbenraar | CC0 1.0 |
| `distracter/idling/fsd50k_eval_396463.wav` | idling | 396463 (eval) | chainsaw_start.wav | Jedo | CC0 1.0 |
| `distracter/idling/fsd50k_eval_406110.wav` | idling | 406110 (eval) | Generator.wav | Yin_Yang_Jake007 | CC0 1.0 |
| `distracter/mosquito/fsd50k_dev_188708.wav` | mosquito | 188708 (dev) | Flying Mosquito.wav | Zywx | CC0 1.0 |
| `distracter/mosquito/fsd50k_dev_98648.wav` | mosquito | 98648 (dev) | 20100606.mosquito.wav | dobroide | CC BY 3.0 |
| `distracter/mosquito/fsd50k_eval_320723.wav` | mosquito | 320723 (eval) | mosquito sound.wav | djfroyd | CC BY 3.0 |
| `distracter/motorcycle/fsd50k_eval_1045.wav` | motorcycle | 1045 (eval) | rbh Harley by 04.wav | RHumphries | CC BY 3.0 |
| `distracter/motorcycle/fsd50k_eval_22615.wav` | motorcycle | 22615 (eval) | bike3.flac | Erdie | CC BY 3.0 |
| `distracter/motorcycle/fsd50k_eval_244503.wav` | motorcycle | 244503 (eval) | motorcycles on the forest highway.wav | Silentwaves | CC BY 3.0 |
| `distracter/mower/fsd50k_dev_156223.wav` | mower | 156223 (dev) | Lawnmower background sound | astrand | CC0 1.0 |
| `distracter/mower/fsd50k_dev_267973.wav` | mower | 267973 (dev) | Lawn Mower.wav | Roulaine | CC BY 3.0 |
| `distracter/mower/fsd50k_dev_97595.wav` | mower | 97595 (dev) | Rotary_Lawnmower_Long.wav | MegMc | CC0 1.0 |
| `distracter/power_tool/fsd50k_eval_128607.wav` | power_tool | 128607 (eval) | Drill_sound.wav | 1pjladd2 | CC0 1.0 |
| `distracter/power_tool/fsd50k_eval_262265.wav` | power_tool | 262265 (eval) | Drill | GowlerMusic | CC BY 3.0 |
| `distracter/power_tool/fsd50k_eval_318203.wav` | power_tool | 318203 (eval) | Drilling into stone.wav | PapercutterJohn | CC0 1.0 |
| `distracter/prop_plane/fsd50k_dev_369482.wav` | prop_plane | 369482 (dev) | Small Aircraft Katana DV20 - outside | Alcappuccino | CC0 1.0 |
| `distracter/prop_plane/fsd50k_dev_403316.wav` | prop_plane | 403316 (dev) | A small propeller plane.wav | straget | CC0 1.0 |
| `distracter/prop_plane/fsd50k_eval_431206.wav` | prop_plane | 431206 (eval) | Airplane Flyby | oscaraudiogeek | CC0 1.0 |
| `distracter/scooter/fsd50k_dev_166717.wav` | scooter | 166717 (dev) | Moped driving by | Bradovic | CC0 1.0 |
| `distracter/scooter/fsd50k_dev_3185.wav` | scooter | 3185 (dev) | PassingMoped01.wav | Pingel | CC0 1.0 |
| `distracter/scooter/fsd50k_dev_55035.wav` | scooter | 55035 (dev) | Scooter.wav | suonidigenova | CC0 1.0 |
| `distracter/shout/fsd50k_eval_102437.wav` | shout | 102437 (eval) | scottstoked_yeehaw.wav | shawshank73 | CC BY 3.0 |
| `distracter/shout/fsd50k_eval_196128.wav` | shout | 196128 (eval) | screechyscreams.ogg | insanity54 | CC0 1.0 |
| `distracter/shout/fsd50k_eval_376638.wav` | shout | 376638 (eval) | Enemy Spotted!/Over There! Voice Lines (Mixed) | EFlexTheSoundDesigner | CC BY 3.0 |
| `distracter/siren/fsd50k_eval_405585.wav` | siren | 405585 (eval) | Siren ( Carpeted Hallway Ambient ).wav | JohnLaVine333 | CC0 1.0 |
| `distracter/siren/fsd50k_eval_405590.wav` | siren | 405590 (eval) | Siren ( Concrete Stairwell Direct ).wav | JohnLaVine333 | CC0 1.0 |
| `distracter/siren/fsd50k_eval_53448.wav` | siren | 53448 (eval) | fire.wav | guitarguy1985 | CC0 1.0 |
| `distracter/thunder/fsd50k_eval_347855.wav` | thunder | 347855 (eval) | thunder boom.wav | cognito perceptu | CC0 1.0 |
| `distracter/thunder/fsd50k_eval_387021.wav` | thunder | 387021 (eval) | Thunder clap | valibuck | CC BY 3.0 |
| `distracter/thunder/fsd50k_eval_397952.wav` | thunder | 397952 (eval) | Thunder Clap And Rumble | Kinoton | CC0 1.0 |
| `distracter/truck/fsd50k_eval_174618.wav` | truck | 174618 (eval) | Truck Starting 001.wav | yottasounds | CC BY 3.0 |
| `distracter/truck/fsd50k_eval_333917.wav` | truck | 333917 (eval) | Parked truck.mp3 | thearxx08 | CC0 1.0 |
| `distracter/truck/fsd50k_eval_52303.wav` | truck | 52303 (eval) | truck reverse.aif | trip2000 | CC0 1.0 |
| `war/air_raid_siren/fsd50k_dev_262278.wav` | air_raid_siren | 262278 (dev) | Warning Siren / War Siren  - Trondheim Norway.mp3 | Ubehag | CC0 1.0 |
| `war/explosion_far/fsd50k_dev_130974.wav` | explosion_far | 130974 (dev) | Large, Low Boom 2.wav | duckduckpony | CC BY 3.0 |
| `war/explosion_far/fsd50k_dev_156499.wav` | explosion_far | 156499 (dev) | Medium Far Explosion | unfa | CC BY 3.0 |
| `war/explosion_far/fsd50k_dev_156500.wav` | explosion_far | 156500 (dev) | Big Far Explosion | unfa | CC BY 3.0 |
| `war/jet/fsd50k_dev_431473.wav` | jet | 431473 (dev) | F35 Jet Low-Flyby.mp3 | johnnythesalesman | CC BY 3.0 |
| `war/jet/fsd50k_eval_397165.wav` | jet | 397165 (eval) | JetFighter01.wav | FullMetalJedi | CC0 1.0 |
| `war/jet/fsd50k_eval_420560.wav` | jet | 420560 (eval) | Fighter Jets Passing Overhead | Nice_Audio | CC0 1.0 |
| `war/machine_gun/fsd50k_dev_101961.wav` | machine_gun | 101961 (dev) | Heavy machine gun.wav | CGEffex | CC BY 3.0 |
| `war/machine_gun/fsd50k_dev_160773.wav` | machine_gun | 160773 (dev) | Maxim Russian machinegun | poissonmort | CC BY 3.0 |
| `war/machine_gun/fsd50k_dev_417688.wav` | machine_gun | 417688 (dev) | LightMachineGun1.wav | SuperPhat | CC0 1.0 |
| `war/mortar_artillery/fsd50k_dev_171475.wav` | mortar_artillery | 171475 (dev) | Artillery (distance) | 2887679652 | CC BY 3.0 |
| `war/mortar_artillery/fsd50k_dev_184275.wav` | mortar_artillery | 184275 (dev) | Distant Tank Shots.flac | qubodup | CC0 1.0 |
| `war/mortar_artillery/fsd50k_dev_187767.wav` | mortar_artillery | 187767 (dev) | Cannon Shot | qubodup | CC0 1.0 |
| `war/mortar_artillery/fsd50k_dev_426161.wav` | mortar_artillery | 426161 (dev) | M224 Mortar Firing Close 03 | Nanashi | CC BY 3.0 |
| `war/mortar_artillery/fsd50k_dev_426165.wav` | mortar_artillery | 426165 (dev) | M224 Mortar Impact  Distant 02 | Nanashi | CC BY 3.0 |
| `war/shout/fsd50k_dev_187294.wav` | shout | 187294 (dev) | "They're braking in" and "we're trapped!" | unfa | CC BY 3.0 |
| `war/shout/fsd50k_dev_362665.wav` | shout | 362665 (dev) | batlle shouts: Come over, this way, hey, over here | 11linda | CC BY 3.0 |
| `war/shout/fsd50k_dev_387829.wav` | shout | 387829 (dev) | medic medic voice | CosmicEmbers | CC BY 3.0 |
| `war/small_arms/fsd50k_eval_174290.wav` | small_arms | 174290 (eval) | REC_1_AT897_STR.wav | ingudios | CC BY 3.0 |
| `war/small_arms/fsd50k_eval_204949.wav` | small_arms | 204949 (eval) | m4-carbine improved.wav | tehlordoswag420 | CC BY 3.0 |
| `war/small_arms/fsd50k_eval_259758.wav` | small_arms | 259758 (eval) | M4 Assault rifle firing.wav | mnslugger20 | CC0 1.0 |
| `war/tank/fsd50k_dev_161886.wav` | tank | 161886 (dev) | WAR-TANK, LEOPARD-ACCELERATION, TURBO-Leopard 2A4 engine acceleration, close by-0002.wav | JoniHeinonen | CC BY 3.0 |
| `war/tank/fsd50k_dev_187676.wav` | tank | 187676 (dev) | Driving Tank Engine.flac | qubodup | CC0 1.0 |
