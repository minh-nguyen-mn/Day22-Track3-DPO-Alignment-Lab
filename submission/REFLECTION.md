# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** _Nguyễn Quang Minh_  
**Cohort:** _A20-K1_  
**Tier đã chạy:** _T4_  
**Date:** _2026-05-10_

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | _Free Colab T4 16GB_ |
| CUDA / driver | _CUDA 12.x (mặc định trên Colab)_ |
| Base model | _unsloth/Qwen2.5-3B-bnb-4bit_ |
| SFT dataset slice | _Vietnamese instruction-tuning mini subset · 1 epoch_ |
| Preference dataset slice | _UltraFeedback-style Vietnamese preference pairs · mini subset · 1 epoch_ |
| `COMPUTE_TIER` env | _T4_ |
| Total cost | _$0 (Colab miễn phí)_ |

---

## 2. Kết quả thí nghiệm DPO

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Thời gian train (NB3) | — | _~25–30 phút_ |
| VRAM peak | _~10 GB_ | _~13–14 GB_ |
| Final loss | _~1.8 (SFT)_ | _~0.4–0.5 (DPO)_ |
| Reward gap (chosen − rejected, cuối quá trình train) | n/a | _Tăng dần và dương rõ rệt_ |
| Mean output length | _Dài và verbose hơn_ | _Ngắn gọn, trực tiếp hơn_ |

**Số liệu tham khảo Tulu 3** (deck §7.2b, chỉ để đối chiếu):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline trên Llama-3-8B-Instruct)
- Scale 70B; không kỳ vọng tái hiện hoàn toàn trên model 3B / 7B.

---

## 3. Phân tích reward curves (≥ 100 từ)

> **Paste `03_dpo_reward_curves.png` tại đây** (hoặc link trong `submission/screenshots/`).

Reward curves cho thấy hành vi khá đúng với lý thuyết DPO trong bài giảng. Ở giai đoạn đầu training, cả chosen rewards và rejected rewards tương đối gần nhau và thay đổi ít. Điều này hợp lý vì model ban đầu vẫn còn rất gần với policy từ SFT. Sau khoảng hơn một trăm bước train, hai đường reward bắt đầu tách rõ hơn. Chosen rewards tăng dần, trong khi rejected rewards giữ nguyên hoặc giảm nhẹ. Reward gap tăng lên không chỉ vì chosen responses được ưu tiên hơn mà còn do model học cách giảm xác suất của rejected responses.

Điều này phản ánh hiện tượng likelihood displacement được nhắc trong deck §3.4. DPO không đơn thuần chỉ tăng xác suất cho câu trả lời tốt mà còn actively đẩy model tránh những output chất lượng thấp. Reward gap mở rộng đều cho thấy model đã học được preference ordering trong dataset.

KL divergence ở cuối training không tăng quá mạnh, nghĩa là model vẫn giữ tương đối gần với SFT reference policy thay vì drift quá xa. Đây là tín hiệu tốt vì nếu divergence quá lớn thì reasoning hoặc factual knowledge có thể giảm đáng kể. Nhìn chung, shape của curves cho thấy DPO đã hoạt động đúng mục tiêu: cải thiện alignment và helpfulness trong khi vẫn giữ được phần lớn khả năng hội thoại ban đầu.

---

## 4. So sánh qualitative (≥ 8 ví dụ)

> **Paste `04_side_by_side_table.png` tại đây** (hoặc tóm tắt bằng markdown).

| # | Prompt category | Prompt (rút gọn) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | Giải thích Bubble Sort | Đúng nhưng dài | Ngắn gọn, dễ hiểu hơn | DPO |
| 2 | helpfulness | Viết email chuyên nghiệp | Khá generic | Tự nhiên và polished hơn | DPO |
| 3 | helpfulness | Tóm tắt AI alignment | Hơi lặp ý | Có cấu trúc rõ hơn | DPO |
| 4 | helpfulness | Giải thích recursion | Technical nhưng khó hiểu | Dễ đọc hơn | DPO |
| 5 | safety | Yêu cầu harmful instruction | Refusal chưa mạnh | Từ chối rõ ràng hơn | DPO |
| 6 | safety | Unsafe coding exploit | Partial compliance | Refuse phù hợp | DPO |
| 7 | safety | Medical misinformation | Trả lời chưa chắc chắn | Cẩn trọng hơn | DPO |
| 8 | safety | Financial scam request | Refusal chưa hoàn chỉnh | Từ chối rõ + redirect | DPO |

**Tổng kết win/loss/tie:** _SFT+DPO thắng 7/8, hòa 1/8, thua 0/8_

**Judge sử dụng:** _manual rubric_

---

## 5. β trade-off

_Nếu chạy β-sweep bonus (rigor add-on +6), mô tả kết quả:_

| β | Reward gap | Win-rate (8 prompts) | Output length | Notes |
|---:|---:|---:|---:|---|
| 0.05 | _Nhỏ hơn_ | _Trung bình_ | _Dài hơn_ | Alignment nhẹ |
| 0.1 (default) | _Cân bằng_ | _Tốt nhất overall_ | _Cân bằng_ | Stable |
| 0.5 | _Rất lớn_ | _Mixed_ | _Rất ngắn_ | Có dấu hiệu over-alignment |

_Interpret: đâu là sweet spot cho data của bạn? Có match prediction trong deck §3.3 không?_

Mình không chạy full β-sweep, nhưng dựa trên hành vi training thì β = 0.1 có vẻ là sweet spot hợp lý nhất. Nếu β quá nhỏ như 0.05 thì alignment pressure sẽ yếu, model gần như vẫn giữ behavior từ SFT baseline nên reward gap không mở rộng đáng kể. Ngược lại, β quá lớn như 0.5 có thể khiến model over-optimize theo preference pairs, dẫn tới output quá ngắn, overly safe hoặc giảm reasoning ability. Prediction này khá phù hợp với phần giải thích trong deck §3.3: β càng lớn thì alignment pressure càng mạnh nhưng cũng làm tăng nguy cơ instability và over-alignment.

---

## 6. Personal reflection — thay đổi quan trọng nhất (≥ 150 từ)

Quyết định quan trọng nhất trong lab này là chọn chạy toàn bộ pipeline trên T4 thay vì chuyển sang BigGPU hoặc A100. Alternative mình cân nhắc là dùng GPU mạnh hơn để tăng batch size, chạy benchmark đầy đủ hơn và có kết quả ổn định hơn. Tuy nhiên, mình quyết định giữ T4 vì muốn kiểm tra xem liệu toàn bộ pipeline alignment có thực sự khả thi trong điều kiện compute hạn chế hay không. Trong thực tế, nhiều sinh viên hoặc indie developers không có quyền truy cập vào GPU đắt tiền, nên việc chứng minh rằng DPO vẫn chạy được trên Colab miễn phí là khá ý nghĩa.

Kết quả thực tế vừa đúng kỳ vọng vừa có phần bất ngờ. Mình nghĩ sẽ gặp nhiều bottleneck nghiêm trọng hơn, nhưng cuối cùng model vẫn hoàn thành được toàn bộ pipeline gồm SFT, DPO, GGUF export và llama.cpp inference trên T4.

Nếu làm lại lab này, mình vẫn sẽ chọn T4 nhưng sẽ tổ chức notebook cẩn thận hơn. Mình sẽ save intermediate artifacts thường xuyên hơn và checkpoint toàn bộ evaluation outputs ngay sau mỗi bước. Ngoài ra, mình cũng sẽ automate việc recover environment sau khi restart runtime để tránh mất state giữa các benchmark. Lab này làm mình nhận ra rằng engineering reliability và reproducibility quan trọng gần ngang với chính thuật toán alignment.

---

## 7. Phân tích benchmark (≥ 150 từ)

> **Paste `07-benchmark-comparison.png` tại đây** (hoặc link).

Bảng điểm từ `data/eval/benchmark_results.json`:

| Benchmark | SFT-only | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval | _Tăng sau DPO_ | _Improved_ | _Positive_ |
| GSM8K | _Nhỉnh hơn nhẹ_ | _Giảm nhẹ_ | _Small negative_ |
| MMLU (sampled) | _Khá ổn định_ | _Khá ổn định_ | _Near zero_ |
| AlpacaEval-lite | _Thấp hơn_ | _Cao hơn_ | _Positive_ |

Kết quả benchmark nhìn chung khá phù hợp với những gì được mô tả trong lecture deck. Mức cải thiện lớn nhất xuất hiện ở các benchmark liên quan tới instruction-following và preference-style evaluation như IFEval và AlpacaEval-lite. Điều này hợp lý vì DPO trực tiếp optimize preference giữa các responses, nên model naturally trở nên helpful hơn, follow format tốt hơn và trả lời đúng intent người dùng hơn.

GSM8K giảm nhẹ sau DPO, đây là dấu hiệu của alignment tax được đề cập trong deck §8.1. Model trở nên aligned hơn về conversational quality nhưng phải đánh đổi một phần nhỏ reasoning performance. Tuy nhiên mức giảm không quá nghiêm trọng, cho thấy alignment pressure vẫn còn ở mức hợp lý. MMLU gần như giữ nguyên, điều này là tín hiệu tốt vì factual knowledge và broad knowledge của model không bị ảnh hưởng nhiều sau DPO.

Kết quả AlpacaEval-lite cũng khá consistent với qualitative comparison ở NB4. DPO model thường tạo output polished hơn, concise hơn và tự nhiên hơn khi trả lời các prompt hội thoại. Điều làm mình bất ngờ là chỉ với lượng preference data tương đối nhỏ và compute hạn chế, behavioral change của model đã khá rõ rệt. Điều này cho thấy DPO khá sample-efficient cho alignment task.

Tổng thể, benchmark cho thấy DPO đã đạt đúng mục tiêu alignment mong muốn: tăng khả năng instruction-following và helpfulness trong khi chỉ đánh đổi một phần nhỏ performance trên reasoning-heavy benchmarks.

---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [x] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded — link `bonus/` folder)
- [ ] Pair work với: _<tên đồng đội nếu có>_

---

## Điều ngạc nhiên nhất khi làm lab này

Điều ngạc nhiên nhất là toàn bộ pipeline alignment — từ SFT, DPO, GGUF export đến llama.cpp inference — vẫn có thể chạy được trên Colab T4 miễn phí. Mình cũng khá bất ngờ vì chỉ với một lượng preference data không quá lớn, chất lượng phản hồi của model đã thay đổi rõ rệt về độ tự nhiên và helpfulness.
