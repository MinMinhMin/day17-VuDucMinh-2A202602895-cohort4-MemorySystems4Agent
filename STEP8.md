# Bước 8 — Phân tích kết quả benchmark

## Output đo được

Hai lần chạy liên tiếp trên trạng thái tạm sạch cho cùng một output:

### Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 1,437 | 16,103 | 0.000 | 0.150 | 0 | 0 |
| Advanced | 1,911 | 27,596 | 1.000 | 1.000 | 345 | 0 |

### Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 374 | 23,273 | 0.000 | 0.150 | 0 | 0 |
| Advanced | 452 | 11,001 | 1.000 | 1.000 | 229 | 4 |

Benchmark chạy offline. Token được ước lượng bằng số ký tự chia 4; `Response quality` là heuristic dùng chung: 85% điểm recall và 15% điểm độ dài câu trả lời gọn. Vì vậy các số này tái lập được trên máy khác, nhưng không phải token usage của provider thật hay điểm chấm từ LLM judge.

## 1. Vì sao Advanced nhớ tốt hơn Baseline?

Ở Standard, Cross-session recall của Baseline là **0.000**, còn Advanced là **1.000**. Stress cũng cho kết quả **0.000 so với 1.000**. Baseline chỉ giữ message theo `thread_id`, còn Advanced dùng `extract_profile_updates()` để ghi facts vào `User.md` theo `user_id`, rồi `_offline_response()` đọc hồ sơ khi câu hỏi được gửi trong thread mới.

## 2. Vì sao Advanced tốn hơn ở hội thoại ngắn?

Trong Standard, Advanced sinh **1,911** agent tokens so với **1,437** của Baseline vì các câu trả lời recall có nội dung facts thay vì chỉ báo chưa biết. Prompt tokens processed là **27,596** so với **16,103**: mỗi lượt Advanced đưa profile vào prompt và có thể ghi thêm facts vào file. Mười hội thoại ngắn chưa kích hoạt compaction, nên phần ngữ cảnh profile chưa được bù bởi lịch sử đã lược bớt.

## 3. Vì sao compact có lợi ở hội thoại dài?

Trong stress, Prompt tokens processed của Advanced là **11,001**, thấp hơn Baseline **23,273** khoảng **52.7%**. `CompactMemoryManager` giữ bốn message gần nhất theo cấu hình mặc định, đưa message cũ vào summary có giới hạn và đã compact **4 lần**. Lợi thế này nằm ở **Prompt tokens processed**; Agent tokens only của Advanced vẫn cao hơn (**452** so với **374**) vì đây là token sinh ra trong câu trả lời, không phải ngữ cảnh được kéo vào prompt.

## 4. Memory tăng trưởng và rủi ro

Advanced tăng **345 bytes** trên Standard và **229 bytes** trên stress; Baseline tăng **0 bytes** vì không có `User.md`. Ở stress có **4 compactions**; Standard có **0**, phù hợp với việc hội thoại ngắn chưa vượt ngưỡng hữu ích để nén. Profile nhỏ ở benchmark này nhưng sẽ tích lũy theo số facts và người dùng trong thời gian dài. Summary compact có thể làm mất chi tiết cũ; một correction được phát biểu rõ nhưng sai cũng có thể ghi đè fact đúng.

## Bonus — xử lý correction mới

Facts được lưu theo field, nên phát biểu hiện tại rõ ràng sẽ thay giá trị cũ của cùng field. Ví dụ, profile chuyển nơi ở từ Huế sang Đà Nẵng trong stress và từ backend engineer sang MLOps engineer trong Standard; câu hỏi ở thread mới nhận lại giá trị hiện tại. Extractor tách mệnh đề đối lập để giữ phần đính chính hiện tại, đồng thời loại giả định tương lai, câu hỏi và đoạn đùa như lời nhắc “product manager”. Hồ sơ dùng ID ổn định có hậu tố băm để hai ID khác nhau không va chạm trên cùng một đường dẫn. Cơ chế này giúp recall correction tốt hơn và giữ một giá trị cho mỗi field thay vì chất chồng các phiên bản cũ, nhưng nếu người dùng đưa ra một correction rõ ràng mà sai thì hệ thống vẫn có thể tin nhầm; memory decay hoặc bước xác nhận sẽ là guardrail tiếp theo.

## Kiểm thử

Toàn bộ **32 kiểm thử** trong `src/` đã chạy thành công, gồm các ca giả định/không chắc chắn, correction sau mệnh đề lịch sử, tách biệt đường dẫn profile, compaction lặp lại và validation dữ liệu đầu vào.
