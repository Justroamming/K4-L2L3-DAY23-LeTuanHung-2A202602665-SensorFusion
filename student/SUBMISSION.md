# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Lê Tuấn Hưng
- MSSV: 2A202602665
- Email: letuanhung2a202602665@student.edu.vn
- Link repo (fork): https://github.com/LeTuanHung/K4-L2L3-DAY23-LeTuanHung-2A202602665-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): 5803f9087a55e799eb2d5aef4dbbd2e1a8066f72

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`:
  - `fusion_mode`: compare
  - `frames`: [0, 198]
  - `segment`: training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
  - `seed`: 0
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`:
  - precision: 0.9701
  - recall: 0.7004
  - tp: 519, fp: 16, fn: 222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`:
  - rmse: 0.1503 m
  - matches: 502
  - sum_sq_err: 11.3436
  - ghost_track_frames: 0
  - missed_gt_frames: 239
  - mean_confirmed_tracks: 2.52
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`:
  - rmse: 0.1359 m
  - matches: 502
  - sum_sq_err: 9.2668
  - ghost_track_frames: 0
  - missed_gt_frames: 239
  - mean_confirmed_tracks: 2.52
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss:
  - Cả hai mode có cùng số `matches=502` và `ghost_track_frames=0`, `missed_gt_frames=239`, nghĩa là số lượng track được ghép và track ma tương đương.
  - Mode `fused` có RMSE thấp hơn (0.1359 m vs 0.1503 m) do camera bổ sung đo 2D giúp tinh chỉnh vị trí track theo chiều ngang (left/up), giảm độ bất định Pxx/Pyy.
  - Camera không tạo track mới cũng không xóa track (chỉ EKF update), nên số liệu vòng đời track (matches, ghosts, misses) giống nhau giữa hai mode.
  - Chênh lệch RMSE = 0.0144 m ≤ 0.05 m thỏa mãn điều kiện nhất quán rubric.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?
   - Lidar cung cấp đo vị trí 3D (x, y, z) trong hệ tọa độ xe → `z` là vector 3×1, `R` là ma trận 3×3 đường chéo với `sigma_lidar_x/y/z² = 0.01`.
   - Camera cung cấp đo pixel 2D (u, v) trên mặt phẳng ảnh → `z` là vector 2×1, `R` là ma trận 2×2 đường chéo với `sigma_cam_i/j² = 25.0`.
   - Lidar dùng H tuyến tính 3×6 (chỉ vị trí, vélôcity = 0); Camera dùng Jacobian 2×6 từ phép chiếu pinhole (chuỗi quy tắc: phép chiếu × xoay xe→camera).

2. Vì sao cần gating Mahalanobis trước khi gán?
   - Khoảng cách Euclid không xét đến độ bất định của track (P) và sensor (R). Mahalanobis chuẩn hóa innovation bởi S = HPHᵀ + R, cho phép loại cặp "gần nhưng không chắc chắn" (innovation lớn so với độ bất định).
   - Chi-square gate với ngưỡng 0.995 loại bỏ các cặp có xác suất giả âm thấp, đảm bảo chỉ gán các cặp hợp lý thống kê.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.
   - Là **track-then-fuse**: LiDAR pass chạy trước (khởi tạo track, gán đo LiDAR, cập nhật score/vòng đời), sau đó Camera pass chỉ refine EKF state cho các track đã có.
   - Log `grade_run.log` cho thấy `mode="lidar"` và `mode="fused"` chạy độc lập với cùng track list, camera không ảnh hưởng `confirmed`, `score`, hay `ghost/miss`.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?
   - Innovation (γ = z - h(x)) sẽ có bias hệ thống: residual không còn phân bố tâm 0 mà dịch chuyển theo hướng lệch calibration (xoay/tiến lùi).
   - Chi-square gate sẽ loại nhiều cặp camera hợp lệ (mhd² tăng), dẫn đến ít update camera hơn, track dựa nhiều hơn vào LiDAR đơn lẻ.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   - Để `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)` biết đây là pass LiDAR hay Camera.
   - Nếu frame LiDAR rỗng (không có đo), track trong FOV vẫn bị trừ score (miss) và có thể bị xóa khi score < threshold. Camera pass không trừ score, không xóa track, không khởi tạo track mới.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.
   - **Xác nhận**: score > `confirmed_threshold` (0.8). Hit LiDAR cộng 1/window, tối đa 1. Miss trong FOV trừ 1/window.
   - **Giữ confirmed sau miss**: Track đã confirmed không bao giờ hạ trạng thái về tentative/initialized, chỉ giảm score.
   - **Xóa track**: (OR) Pxx > max_P (9.0) HOẶC Pyy > max_P HOẶC (confirmed VÀ score < delete_threshold 0.6) HOẶC (chưa confirmed VÀ score ≤ 0). Camera pass không bao giờ xóa track.

## Bonus (không bắt buộc)

- Không

## Khai báo sử dụng AI (bắt buộc)

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Kilo Code (nvidia/nemotron-3-ultra-550b-a55b:free)
- Dùng cho phần nào (hàm, câu hỏi, debug):
  - Implement Part E (kalman.py): build_F, build_Q, ekf_predict, innovation, innovation_covariance, ekf_update
  - Implement Part G (camera_fusion.py): is_in_field_of_view, camera_measurement_prediction, build_camera_measurement
  - Implement Part F (association.py): mahalanobis_distance, chi2_gate, association_cost_matrix, pick_next_pair, associate_and_update
  - Implement Part H (track_management.py): init_track_state_from_meas, update_track_score, should_delete_track
  - Chạy test pytest và fix lỗi dựa trên assertion message
  - Chạy fusion-run-lab và kiểm tra artifacts
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức):
  - pytest student/tests/test_kalman.py -v (4/4 passed)
  - pytest student/tests -k "camera" -v (21 passed)
  - pytest student/tests -k "mahalanobis or greedy or out_of_fov or association" -v (9 passed)
  - pytest student/tests -k "lifecycle or score or deletion or delete" -v (22 passed)
  - Toàn bộ test suite: 110 passed, 0 failed, 0 xfailed
  - fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0 → 6 artifacts sinh ra đúng
  - Kiểm tra metrics.json, grade_run.log, metrics_lidar.json, metrics_fused.json, grade_run_lidar.log, grade_run_fused.log
  - Đối chiếu RMSE, matches, ghost_track_frames, missed_gt_frames giữa hai mode

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))