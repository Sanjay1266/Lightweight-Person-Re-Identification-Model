# Dataset Protocol & Ground-Truth Annotation Audit Report

**Reference Paper**: DCR-ReID (IEEE Transactions on Circuits and Systems for Video Technology, 2023)
**Date**: 2026-10-28 21:12:00 IST

---

## 1. Official CC-ReID Datasets in the Paper

The DCR-ReID paper validates its methodology on three official Cloth-Changing Person Re-Identification benchmarks:
1. **PRCC (Person Re-ID under Clothing Change)**: Image-based; 221 identities, 3 cameras (A: Query, B: Same Clothes, C: Cloth Changing). Available locally in this repository.
2. **LTCC (Long-Term Cloth-Changing)**: Image-based; 152 identities, 12 cameras, up to 14 outfits per person. (Not bundled locally in this repository).
3. **CCVID (Cloth-Changing Video Person Re-ID)**: Video-based; 226 identities, 2,856 tracklet sequences, 347,833 frames. (Not bundled locally in this repository).

---

## 2. PRCC Dataset Verification & Ground-Truth Protocol

- **Local Path**: `Data_set/prcc/rgb/`
- **Train Split**: `train/` containing 150 identity directories (`'092'`, `'095'`, ..., `'332'`), totalling **17,896 images**.
- **Test Split**: `test/` containing 71 identity directories (`'001'`, `'002'`, ..., `'328'`), totalling **10,800 images**.
- **Zero Identity Leakage Verified**:
  $$\text{Train PIDs} \cap \text{Test PIDs} = \emptyset \quad (\text{Exactly 0 overlapping identities})$$

### Camera Role & Ground-Truth Clothing Conditions
- **Camera A (`camid = 0`)**: Query view (3,384 images across 71 test identities). Subject in **Outfit 1**.
- **Camera B (`camid = 1`)**: Gallery view (3,873 images across 71 test identities). Subject in **Outfit 1 (Same Clothes Condition - SC)**.
- **Camera C (`camid = 2`)**: Gallery view (3,543 images across 71 test identities). Subject in **Outfit 2 (Cloth-Changing Condition - CC)**.

### Official PRCC Evaluation Settings:
1. **`prcc_cc` (Cloth-Changing Evaluation)**:
   - Query: Camera A (3,384 images)
   - Gallery: Camera C (3,543 images)
   - Evaluates pure invariant biometric retrieval across physical apparel changes.
   - Paper Reported Target: **57.2% CMC@1, 57.4% mAP** (with ResNet-50 at 384×192).
2. **`prcc_sc` (Same-Clothes Evaluation)**:
   - Query: Camera A (3,384 images)
   - Gallery: Camera B (3,873 images)
   - Evaluates general surveillance retrieval without clothing variation.
   - Paper Reported Target: **85.4% CMC@1, 83.6% mAP**.
3. **`prcc_all` (Combined / General Setting)**:
   - Query: Camera A (3,384 images)
   - Gallery: Camera B + Camera C (7,416 images)
   - Paper Reported Target: **72.8% CMC@1, 68.5% mAP**.

---

## 3. Custom Repository Subsets Audit (`both_small`, `with_bag`, `without_bag`, `both_large`)

In addition to PRCC, the repository includes four surveillance splits formatted following the Market-1501 convention (`bounding_box_train`, `bounding_box_test`, `query` with filename pattern `XXXX_cY_fZ.jpg`):

1. **`both_small`** (2,919 train / 2,146 gallery / 475 query): Combined small campus surveillance split with 250 identities.
2. **`with_bag`** (1,157 train / 1,131 gallery / 322 query): Subset of 82 identities carrying backpacks, shoulder bags, or accessories.
3. **`without_bag`** (1,770 train / 986 gallery / 179 query): Subset of 167 identities without occluding bags.
4. **`both_large`** (11,719 train / 10,048 gallery / 1,265 query): Large-scale 250-identity surveillance gallery benchmark.

### Crucial Distinction from Official Benchmarks:
- These four subsets **are not official DCR-ReID cloth-changing benchmarks**. They represent viewpoint, bag occlusion, and scale variations under traditional surveillance conditions.
- They must not be conflated with the paper's reported PRCC/LTCC benchmarks.
- In this repository, they are used to validate edge robustness against accessories and occlusions under the Lightweight Track B.
