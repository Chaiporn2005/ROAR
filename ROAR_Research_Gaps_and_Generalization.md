# ROAR: Research Gaps ในงานปัจจุบัน และแนวทางขยายไปยัง Experiment B / Experiment C

**โปรเจกต์:** Metabolise — Metabolomics-based NCD risk prediction
**ผู้จัดทำ:** nut, KMITL AI Engineering
**อ้างอิง:** `github.com/Chaiporn2005/ROAR` (Experiment A) และ `github.com/Chaiporn2005/Forecast/Bridge_ai_summit_nmr_2026` (Experiment A/B/C)

เอกสารนี้วิเคราะห์ 2 เรื่องตามที่ขอ: (1) ROAR ที่ทำเสร็จแล้วบน Experiment A มี research gap อะไรบ้าง โดยอิงจากผลลัพธ์จริงในรีโป ไม่ใช่ทฤษฎีทั่วไป และ (2) ถ้าจะเอา ROAR ไปจับ Experiment B (NHANES clinical) และ Experiment C (MTBLS1 urine NMR) จะต้องปรับอะไร และจะเจอ gap อะไรเพิ่มที่ Experiment A ไม่เจอ

---

## ส่วนที่ 1 — Research gap ใน ROAR บน Experiment A (MTBLS242 obesity)

สรุปโปรโตคอลปัจจุบัน: n=177 (preop 106 / 12-month 71), 21 metabolites, baseline PR-AUC 0.9718 ± 0.0242, ทำ cumulative top-k removal ที่ k = 1, 3, 5, 8, 11 โดยเทียบ informed removal (ตาม SHAP ranking คงที่จาก baseline model) กับ random removal 30 draws/k แล้ววัด percentile ของ informed score ในการกระจายของ random

### Gap ที่ยืนยันได้จากข้อมูลจริง (มีหลักฐานในรีโป)

**1. Feature redundancy / proxy effect — เกิดขึ้นจริง ไม่ใช่แค่ทฤษฎี**

`proxy_detection.csv` แสดงว่าหลังตัด top features ออกแล้ว retrain ใหม่ โมเดลไปยึด feature อื่นที่ correlate สูงกับของที่ถูกตัดทันที:

| k ที่ตัด | Feature ใหม่ที่ขึ้นมาเป็นอันดับ 1 หลัง retrain | Correlate กับ feature ที่ถูกตัด | Pearson r |
|---|---|---|---|
| 1 | isopropanol | Dimethyl sulfone | −0.42 |
| 3 | L-valine | Dimethyl sulfone | −0.46 |
| 8, 11 | D-phenylalanine | L-valine | **0.57** |

ที่ k=8 และ 11 คือจุดที่ตัด L-valine (biology marker ตัวหลัก) ออกไปแล้ว โมเดลก็ไปเกาะ D-phenylalanine แทนทันที (r=0.57) — นี่คือหลักฐานตรงว่า ROAR วัด "โมเดลพึ่งพาสัญญาณกลุ่มไหน" ได้ แต่ไม่ได้บอกว่าฟีเจอร์ที่เหลืออยู่เป็น "สาเหตุ" จริงหรือเป็นแค่ตัวแทนที่สหสัมพันธ์กันเฉยๆ ในข้อมูล metabolomics ที่ metabolite หลายตัวอยู่ใน pathway เดียวกัน ปัญหานี้จะรุนแรงกว่าข้อมูล tabular ทั่วไปโดยธรรมชาติของข้อมูลเอง

**2. Static feature selection — เป็น gap จริงและเชื่อมกับข้อ 1**

Ranking ที่ใช้ตัดฟีเจอร์ (`baseline_shap_ranking.csv`) คำนวณครั้งเดียวจาก baseline model แล้วใช้ลำดับเดิมตัดสะสมไปเรื่อยๆ ทั้งที่ proxy_detection.csv ชี้ว่าพอตัดไปถึง k=8 โมเดลที่ retrain แล้วไม่ได้พึ่ง ranking เดิมอีกต่อไป (มันเปลี่ยนไปพึ่ง D-phenylalanine ซึ่งอยู่อันดับ 15 ใน ranking เดิม) การที่ยังตัดตามลำดับเดิม (rank 9, 10, 11 = Pyruvic acid, acetoacetate, citrate) แทนที่จะตัด D-phenylalanine ที่โมเดลใหม่พึ่งจริง ทำให้ ROAR curve ที่ k สูงๆ ไม่ได้สะท้อน "โมเดลตอนนั้น" พึ่งอะไรอยู่จริง — ตรงกับ "Recursive ROAR" ในบทความที่ส่งมา คือจุดที่ยังไม่ได้ทำ

**3. Epistemic gap เฉพาะโปรเจกต์นี้ — ROAR แยกไม่ออกระหว่าง "artifact ที่โมเดลพึ่งจริง" กับ "biology ที่โมเดลพึ่งจริง"**

นี่คือ finding ที่สำคัญที่สุดของโปรเจกต์ และเป็น gap ที่ไม่อยู่ใน 4 หมวดที่บทความส่งมาพูดถึงเลย จาก `contrast_artifact_vs_biology.csv`:

| กลุ่มที่ตัดออก | Feature | PR-AUC หลังตัด | Percentile ในกลุ่ม random (k=3) |
|---|---|---|---|
| known artifact trio | isopropanol, methanol, Dimethyl sulfone | 0.9523 | 3.3% (แย่กว่า random 29/30 ครั้ง) |
| top literature biology trio | lipoproteins, L-valine, L-tyrosine | 0.9605 | 6.7% (แย่กว่า random 28/30 ครั้ง) |

ทั้งสองกลุ่มมี faithfulness score ที่ ROAR วัดได้ "ใกล้เคียงกันมาก" (ต่างกันแค่ ~0.008 PR-AUC และ 3.3 percentile point) ทั้งที่กลุ่มแรกคือสิ่งปนเปื้อนจากขั้นตอนผ่าตัด (isopropanol = สารทำความสะอาดผิวหนัง) ไม่ใช่ชีววิทยาของโรคอ้วนเลย สรุปคือ **ROAR บอกได้แค่ว่าโมเดล "พึ่งพา" (dependence) ฟีเจอร์กลุ่มไหนจริง แต่บอกไม่ได้ว่าสิ่งที่พึ่งพานั้น "ถูกต้องทางชีววิทยา" (validity) หรือเปล่า** — เป็นข้อจำกัดเชิงญาณวิทยา (epistemic) ของตัว metric เอง ไม่ใช่ implementation bug ต้องอาศัยความรู้โดเมน (เหมือนที่ทำไว้แล้วในสไลด์) มาช่วยตีความคู่กับตัวเลข ROAR เสมอ ใช้ ROAR เดี่ยวๆ ไม่พอ

**4. Statistical resolution — ข้อจำกัดจาก computational cost**

Percentile ที่รายงาน (0.0%, 3.3%, 6.7%) มาจากการเทียบกับ random draws แค่ 30 ครั้งต่อ k เท่านั้น (`n_random_draws_per_k: 30`) ดังนั้นความละเอียดที่วัดได้ที่ดีที่สุดคือ 1/30 ≈ 3.3 percentage point — ตัวเลข "percentile 0.0" ไม่ได้แปลว่า p-value = 0 จริงๆ แปลว่า "แย่กว่าทุกตัวใน 30 ตัวอย่างที่สุ่มมา" เท่านั้น การจะได้ p-value ที่ละเอียดกว่านี้ (เช่น 1/1000) ต้อง retrain เพิ่มอีกหลายสิบเท่า ซึ่งตรงกับ "Computational Cost" ในบทความที่ส่งมาโดยตรง — ในโปรเจกต์นี้ยังจัดการได้เพราะ n=177 และ 21 features เล็ก (runtime รวม 95.9 วินาที) แต่เป็น gap ที่จะโตขึ้นทันทีถ้าข้อมูลใหญ่ขึ้นหรือฟีเจอร์เยอะขึ้น (ดูส่วน Exp C ด้านล่าง)

### Gap ที่ "ไม่ใช่ปัญหา" ในงานนี้ (ออกแบบมาป้องกันไว้แล้ว)

**Masking Artifacts / OOD (Gap 3 ในบทความที่ส่งมา) — ไม่เกิดในงานนี้ เพราะออกแบบเลี่ยงไว้แต่แรก**

บทความที่ส่งมาพูดถึงปัญหาของ perturbation-based XAI evaluation ที่แทนค่าฟีเจอร์ที่ตัดด้วย mask token หรือค่าคงที่ ทำให้ input หลุด distribution เดิม (out-of-distribution) — แต่การ implement ROAR ในโปรเจกต์นี้ **ตัดทั้ง column ออกจาก dataframe แล้ว retrain โมเดลใหม่ทั้งชุด** (ไม่ใช่แทนด้วย mask/mean แล้วรันโมเดลเดิม) ดังนั้นไม่มีปัญหา input หลุด distribution เลย — นี่คือข้อดีของ "true ROAR" เทียบกับ ROAD/GOAR variant ที่บทความพูดถึง จุดนี้ไม่ต้องแก้ไขอะไรเพิ่ม

### สรุปด่วน (สิ่งที่ควรทำต่อถ้าจะปิด gap ของ Exp A)

- เพิ่ม recursive re-ranking ที่ทุก checkpoint (ไม่ต้องทุกฟีเจอร์ก็ได้ เพราะมีแค่ 21 ตัว, cost ยังรับได้) → ปิด gap ข้อ 2
- รายงาน proxy correlation ควบคู่กับทุก checkpoint ใน ROAR curve ไม่ใช่แค่ตาราง proxy_detection แยกต่างหาก → ทำให้ gap ข้อ 1 มองเห็นง่ายขึ้นตอนอ่านกราฟหลัก
- ระบุชัดในรายงาน/thesis ว่า contrast_artifact_vs_biology คือหลักฐานของ "epistemic limitation" ของ ROAR เอง ไม่ใช่แค่ผลข้างเคียงที่น่าสนใจ — นี่คือ contribution เชิงวิธีวิทยาที่มีคุณค่าที่สุดของงานนี้
- ถ้าอยากได้ percentile ละเอียดกว่านี้ ลองเพิ่ม random draws (เช่น 30 → 100) เฉพาะที่ k=1 กับ k=3 ซึ่ง cost ยังต่ำ แทนที่จะเพิ่มทุก k

---

## ส่วนที่ 2 — เอา ROAR ไปจับ Experiment B (T2DM, NHANES clinical)

### ข้อมูลจริงของ Exp B (จาก README)

n = 6,113 (NHANES 2013–14), prevalence 14.3%, 13 ฟีเจอร์คลินิกทั่วไป (ไม่รวม HbA1c/glucose เพราะเป็นตัวกำหนด label), PR-AUC 0.416 ± 0.030 (เทียบ baseline การทายมั่วตาม prevalence 0.143 → lift ~2.9 เท่า), ROC-AUC ≈ 0.83, constrained 11/13 ฟีเจอร์

SHAP ranking: Age (1.03) > Triglycerides (0.47) > Total cholesterol (0.31) > Systolic BP (0.26) ≈ Albumin (0.26) > Race (0.25) > HDL (0.17) ≈ Creatinine (0.17) > BUN (0.11) > Diastolic BP, TC/HDL, Uric acid, Sex (≤0.08)

### ทำไมงานนี้ "ง่ายกว่า" Exp A ในมุม ROAR

จำนวนฟีเจอร์น้อยกว่า (13 vs 21) และเป็นข้อมูล clinical ทั่วไปที่ train เร็ว แม้ n จะใหญ่กว่ามาก (6,113 vs 177) — cost ของการ retrain ซ้ำๆ ต่อ checkpoint ต่อ random draw ยังจัดการได้สบาย ไม่ใช่ bottleneck เหมือนที่กังวลไว้ในบทความที่ส่งมา เพราะ XGBoost บน tabular data 13 คอลัมน์เทรนเร็วมากไม่ว่า n จะเท่าไหร่

### แต่มี gap ใหม่ที่ Exp A ไม่เจอ

**ไม่มี "known artifact" ให้เทียบแบบ Exp A** — จุดแข็งที่สุดของ ROAR ใน Exp A (contrast_artifact_vs_biology) มาจากการที่ dataset มี exogenous contaminant ที่รู้แน่ชัด (isopropanol/methanol จากขั้นตอนผ่าตัด) ให้ทดสอบเทียบกับ biology จริง แต่ฟีเจอร์ทั้ง 13 ตัวใน NHANES ล้วนเป็นตัวชี้วัดทางสรีรวิทยาที่มีความหมายทางคลินิกทั้งหมด (age, BP, lipids, renal markers) — ไม่มีตัวไหนเป็น "สิ่งปนเปื้อนที่รู้แน่ชัดว่าไม่เกี่ยวกับโรค" เหมือน isopropanol ดังนั้น ROAR บน Exp B จะบอกได้แค่ "โมเดลพึ่งอะไรจริง" (functional dependence) แต่ **ทำ epistemic contrast แบบ Exp A ไม่ได้เลย นอกจากจะจงใจใส่ negative-control feature ปลอมเข้าไปในชุดข้อมูล** (เช่น ฟีเจอร์สุ่มที่ไม่เกี่ยวกับผู้ป่วยเลย แล้วดูว่า SHAP จัดอันดับมันต่ำจริงไหม เป็นวิธี sanity-check แทน)

**Feature redundancy น่าจะรุนแรงกว่า Exp A** — จากรายชื่อฟีเจอร์ มีคู่ที่ collinear กันชัดเจนอยู่แล้วโดยธรรมชาติ: Total cholesterol กับ HDL กับ TC/HDL ratio (ตัวหลังคำนวณมาจากสองตัวแรกโดยตรง) และ Creatinine กับ BUN (ทั้งคู่เป็น renal marker) คาดได้เลยว่าถ้าตัด Total cholesterol ออก โมเดลน่าจะไปพึ่ง TC/HDL หรือ HDL แทนทันที — นี่คือจุดที่ ROAR + proxy detection บน Exp B น่าจะให้ insight ที่ชัดกว่า Exp A ด้วยซ้ำ เพราะรู้ mechanism ของ collinearity ล่วงหน้าอยู่แล้ว ทำให้ตรวจสอบผลได้ตรงประเด็น

**แผนปรับ ROAR สำหรับ Exp B (ข้อเสนอ):**
คงโครงเดิม (informed cumulative removal vs random removal, retrain XGBoost ใหม่ทุกครั้ง, 5-fold CV) แต่ปรับ checkpoint ตามจำนวนฟีเจอร์ที่น้อยกว่า เช่น k = 1 (Age), 2 (+Triglycerides), 4 (+Total cholesterol, +Systolic BP), 6 (+Albumin, +Race), 9 (+HDL, +Creatinine, +BUN) เนื่องจากคำนวณเร็ว แนะนำทำ **recursive re-ranking ทุก checkpoint** ได้เลย (ปิด gap ข้อ 2 ของ Exp A ไปด้วยในตัว) และเพิ่ม sanity-check ด้วย negative-control feature เพื่อชดเชยการไม่มี known-artifact ตามธรรมชาติ

---

## ส่วนที่ 3 — เอา ROAR ไปจับ Experiment C (T2DM, MTBLS1 urine NMR)

### ข้อมูลจริงของ Exp C (จาก README + POSTER.md)

132 samples × 220 NMR variables (115 identified metabolites + 105 unassigned chemical-shift buckets), 48 diabetes vs 84 control จาก 42 คน (12 healthy × 7 ตัวอย่าง + 30 T2DM × 1–3 ตัวอย่าง) ใช้ **StratifiedGroupKFold แบบ subject-grouped** (ไม่ใช่ stratified ธรรมดา) เพราะแต่ละคนมีหลายตัวอย่างซ้ำ ผล ROC-AUC = 0.976 ± 0.022 (เทียบ naive-stratified 0.986 → leakage gap เล็กน้อย 0.010) constrained 32/220 ตัวแปร

SHAP top drivers: 2-oxoisovalerate, isoleucine (BCAA/keto-acid), aromatic buckets 7.3–7.9 ppm (คาดว่าคือ hippurate), N-methylnicotinamide/nicotinate, allantoin, fumarate, histidine, **ethanol (ระบุไว้ชัดเจนใน POSTER.md ว่าเป็น exogenous confounder จากอาหาร/การจัดเก็บตัวอย่าง)**

**Known limitation ที่ระบุไว้ในโปรเจกต์เอง:** ตัวอย่าง control ทุกตัวขึ้นต้นด้วย `ADG19007u` และตัวอย่าง diabetes ทุกตัวขึ้นต้นด้วย `ADG10003u` — แปลว่า **label ทายได้ 100% จาก acquisition batch อย่างเดียว** subject-grouped CV ป้องกันแค่ leakage ระดับ "คนเดียวกัน" ข้าม fold แต่ป้องกัน batch-level leakage ไม่ได้เลย เพราะทุก subject ในกลุ่ม control อยู่ batch เดียว และทุก subject กลุ่มเบาหวานอยู่อีก batch หนึ่ง ทำให้ผู้จัดทำเองสรุปไว้ตรงๆ ว่า "PR-AUC/ROC-AUC นี้เป็นเพดานสูงสุด (ceiling) ไม่ใช่หลักฐานยืนยัน biology จริง"

### นี่คือ dataset ที่เหมาะกับ ROAR's artifact-vs-biology contrast มากที่สุด — มากกว่า Exp A ด้วยซ้ำ

Exp C มี "ethanol" เป็น known-exogenous-confounder ที่ระบุไว้แล้วในผลลัพธ์ SHAP เหมือนกับ isopropanol/methanol ใน Exp A ทุกประการ ข้อเสนอที่เป็นรูปธรรม: ทำ contrast_artifact_vs_biology.csv เวอร์ชัน Exp C โดยตัด **ethanol** ออกเทียบกับตัด **BCAA/keto-acid trio (2-oxoisovalerate, isoleucine, + อีกหนึ่งตัวจาก hippurate region)** ออก แล้วดูว่า percentile-in-random-distribution ต่างกันแค่ไหน — ถ้าออกมาใกล้เคียงกันเหมือน Exp A ก็จะเป็นหลักฐานชิ้นที่สองที่สนับสนุน epistemic gap เดิม (ROAR แยก artifact จาก biology ไม่ออก) แต่ที่น่าสนใจกว่านั้นคือ Exp C ยังมี **batch confound ที่ perfectly correlate กับ label ทั้งชุด** ซึ่งรุนแรงกว่า Exp A มาก (ethanol เป็นแค่ 1 ใน 220 ฟีเจอร์ที่ SHAP จัดอันดับมาไม่สูงสุด ในขณะที่ batch คือตัวแปรแฝงที่ปนอยู่ในทุกฟีเจอร์พร้อมกันเป็นระบบ) — ROAR (ซึ่งตัดทีละฟีเจอร์/กลุ่มฟีเจอร์) **ตรวจจับ systematic batch confound แบบนี้ไม่ได้เลยไม่ว่าจะตัดฟีเจอร์ไหนออก** เพราะสัญญาณ batch อาจกระจายอยู่ในหลายสิบฟีเจอร์พร้อมกันในระดับที่อ่อนเกินกว่าจะเห็นจาก top-k ranking — เป็น**ข้อจำกัดใหม่ของ ROAR ที่ Exp A ไม่เคยเจอ**: ROAR ทดสอบ dependence ต่อฟีเจอร์ที่ระบุได้ทีละตัว/กลุ่มเล็กๆ แต่ตรวจจับ confound ที่กระจายตัวข้ามฟีเจอร์จำนวนมาก (diffuse confound) ไม่ได้

### Gap อื่นที่รุนแรงขึ้นเมื่อ scale ไป Exp C

**1. Computational cost กลายเป็นปัญหาจริง (ตรงกับ Gap 1 ในบทความที่ส่งมา)**
220 ฟีเจอร์ เทียบกับ 21 ใน Exp A คือมากกว่า 10 เท่า ถ้าจะทำ cumulative removal + random draws (30 ครั้ง/checkpoint) + retrain แบบ subject-grouped CV เหมือนเดิมทุกจุด จำนวน retrain รวมจะสูงขึ้นมาก และช่องว่างของ "จำนวนฟีเจอร์ที่เป็นไปได้ในการสุ่ม" ที่ k ใดๆ ก็โตขึ้นมหาศาล (C(220,k) เทียบ C(21,k)) แปลว่า 30 random draws ต่อ k จะให้ความละเอียดของ percentile ที่หยาบกว่าความเป็นจริงมาก ข้อเสนอ: ลด scope เหลือเฉพาะ 115 identified metabolites ก่อน (ตัด 105 unassigned buckets ออกจาก pool การสุ่ม แต่ยังเก็บไว้เป็น covariate) หรือ group ฟีเจอร์เป็น pathway-level ก่อนทำ ROAR แทนที่จะทำระดับ raw variable

**2. ต้องคง subject-grouped CV logic ไว้ทุกรอบ retrain อย่างเคร่งครัด**
ถ้า implement ROAR loop แบบ Exp A ตรงๆ (ที่ใช้ stratified k-fold ธรรมดา) แล้วลืมย้าย StratifiedGroupKFold + การ reconstruct subject จาก consecutive sample-number blocks มาด้วย จะทำให้ same-person leakage กลับมาแทรกในทุก retrain ของ ROAR curve — ผลที่ได้จะดูดีเกินจริงและ invalid ทันที ไม่ใช่แค่รายละเอียดทางเทคนิค แต่เป็น precondition ที่ทำให้ผลทั้งชุดใช้ได้หรือใช้ไม่ได้เลย

**3. Feature redundancy รุนแรงกว่า Exp A โดยธรรมชาติของข้อมูล NMR**
ใน NMR หนึ่ง metabolite สามารถให้สัญญาณ (peak) ได้หลายตำแหน่งในสเปกตรัม ดังนั้นในบรรดา 220 ตัวแปร มีความเป็นไปได้สูงว่าหลายคอลัมน์คือ "peak คนละตำแหน่งของสารตัวเดียวกัน" ต่างจาก Exp A ที่ 21 metabolites ถูกระบุชื่อแยกจากกันชัดเจนแล้ว การตัด "2-oxoisovalerate" ออกอาจทำให้โมเดลไปเกาะ peak อื่นของสารตัวเดียวกันทันที ซึ่ง proxy_detection ใน Exp A ตรวจจับปรากฏการณ์แบบนี้ได้ในระดับ metabolite-to-metabolite (r=0.57 ระหว่าง L-valine กับ D-phenylalanine ซึ่งเป็นคนละสาร) — ใน Exp C ปัญหาจะลึกกว่านั้นคือ same-molecule multi-peak redundancy ซึ่ง proxy detection แบบเดิมจะตรวจจับได้ (correlation สูงมาก) แต่การตีความจะยากกว่า เพราะ 105 ใน 220 ตัวแปรไม่มีชื่อสารเคมีให้ตรวจสอบกับวรรณกรรมได้เลย — ถ้า ROAR ชี้ว่าโมเดลไปเกาะ unassigned bucket ตัวหนึ่งหลัง retrain จะไม่มีทางรู้ได้ว่ามันคือ biology จริง หรือ artifact หรือแค่ peak ซ้ำของสารที่ถูกตัดไปแล้ว — เป็นทางตันเชิงการตีความที่ Exp A ไม่เจอเพราะทุก metabolite มีชื่อ

**4. ขนาดตัวอย่างเล็กมากในมิติของ subject**
42 คน (26 subject-groups ตามที่ README ระบุ) คือขีดจำกัดล่างของความละเอียดสถิติที่ทำได้ ไม่ว่าจะสุ่ม random draws กี่ครั้งก็ตาม เพราะจำนวนกลุ่มที่ไม่ซ้ำกันมีจำกัดโดยพื้นฐาน — percentile-based faithfulness test จะมี variance สูงกว่า Exp A (n=177 ไม่มี subject-grouping) มาก ควรรายงาน confidence interval ของ percentile ด้วยไม่ใช่แค่ตัวเลขเดียว

### แผนปรับ ROAR สำหรับ Exp C (ข้อเสนอสรุป)

1. จำกัด pool ของฟีเจอร์ที่ใช้ทำ informed/random removal ให้เหลือ 115 identified metabolites ก่อน (ลด combinatorial cost และแก้ปัญหาตีความ unassigned bucket)
2. คง StratifiedGroupKFold + subject-reconstruction logic เดิมไว้ในทุกรอบ retrain ของ ROAR
3. ทำ contrast แบบ Exp A แต่ใช้ ethanol เป็น known-artifact arm เทียบกับ BCAA/keto-acid trio เป็น known-biology arm — นี่คือจุดที่ให้คุณค่าทางวิชาการสูงสุด เพราะจะเป็นการ replicate finding หลักของ Exp A ในอีก dataset หนึ่งที่ modality ต่างกันโดยสิ้นเชิง (NMR urine vs NMR serum, T2DM vs obesity)
4. รายงานผลพร้อมข้อจำกัดตรงๆ ว่า ROAR ตรวจจับ batch confound ที่กระจายทั่วทุกฟีเจอร์ไม่ได้ — ผลลัพธ์ ROAR ของ Exp C ทั้งหมดต้องอ่านคู่กับ known limitation เรื่อง batch ที่ระบุไว้ใน README เสมอ ไม่สามารถใช้ ROAR มา "กู้" ความน่าเชื่อถือของโมเดลจาก batch confound นี้ได้

---

## ตารางสรุปเปรียบเทียบทั้ง 3 การทดลอง

| ประเด็น | Exp A (obesity, MTBLS242) | Exp B (T2DM, NHANES) | Exp C (T2DM, MTBLS1 NMR) |
|---|---|---|---|
| n / features | 177 / 21 | 6,113 / 13 | 132 (42 คน) / 220 |
| CV scheme | stratified 5-fold | stratified 5-fold | **subject-grouped** 5-fold |
| Known exogenous artifact ให้ทดสอบ | มี (isopropanol, methanol, DMSO) | **ไม่มี** — ต้องใส่ negative control เอง | มี (ethanol) |
| Computational cost ของ ROAR | ต่ำ (~96 วิ) | ต่ำมาก (features น้อย) | **สูง** (220 features, ต้อง subject-group ทุก retrain) |
| ความเสี่ยง feature redundancy | ปานกลาง (ยืนยันแล้ว, r สูงสุด 0.57) | สูง (TC/HDL/TC-ratio, Creatinine/BUN collinear โดยโครงสร้าง) | **สูงสุด** (multi-peak same-molecule redundancy + ตีความไม่ได้สำหรับ unassigned buckets) |
| Diffuse/systemic confound ที่ ROAR มองไม่เห็น | ไม่มีที่ทราบ | ไม่มีที่ทราบ | **มี** — acquisition batch สัมพันธ์กับ label 100% |
| ความพร้อมสำหรับ recursive ROAR | พร้อม (cost รับได้) | พร้อมที่สุด (cost ต่ำสุด) | ต้อง trade-off (ทำระดับ checkpoint ไม่ใช่ทุกฟีเจอร์) |

---

### เอกสารอ้างอิงข้อมูลที่ใช้ในการวิเคราะห์นี้

- `Chaiporn2005/ROAR/results/roar_curve.csv`, `contrast_artifact_vs_biology.csv`, `proxy_detection.csv`, `protocol.json`, `baseline_shap_ranking.csv`
- `Chaiporn2005/Forecast/Bridge_ai_summit_nmr_2026/README.md`
- `Chaiporn2005/Forecast/Bridge_ai_summit_nmr_2026/poster_figures/POSTER.md`
