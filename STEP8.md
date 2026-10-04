# Bước 8 — Nhận xét kết quả benchmark

Mình chạy benchmark hai lần với state tạm mới và nhận được cùng một kết quả. Các bảng dưới đây là số liệu từ chế độ offline.

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

Token ở đây được ước lượng từ độ dài văn bản, khoảng 4 ký tự cho một token. `Response quality` cũng là điểm heuristic: 85% dựa trên recall và 15% dựa trên độ dài câu trả lời. Vì vậy benchmark dễ chạy lại và so sánh, nhưng không đại diện cho token usage thật của từng provider hay điểm đánh giá từ LLM judge.

## Khả năng nhớ qua các thread

Ở cả hai bộ dữ liệu, Baseline đạt recall **0.000**, còn Advanced đạt **1.000**. Đây là khác biệt đúng với thiết kế: Baseline chỉ giữ lịch sử theo `thread_id`; Advanced lưu các thông tin ổn định vào `User.md` theo `user_id`, rồi dùng hồ sơ đó khi câu hỏi được gửi ở thread mới.

## Chi phí ở hội thoại ngắn

Trong Standard, Advanced xử lý **27,596 prompt tokens**, cao hơn **16,103** của Baseline. Advanced đưa hồ sơ vào ngữ cảnh và trả lời recall bằng thông tin cụ thể, nên tốn thêm token. Các hội thoại trong bộ Standard chưa đủ dài để kích hoạt compact; ở quy mô này, lợi ích nhớ lâu chưa bù được phần chi phí tăng thêm.

## Tác dụng của compact ở hội thoại dài

Trong stress test, Advanced xử lý **11,001 prompt tokens**, so với **23,273** của Baseline — giảm khoảng **52.7%**. Hệ thống compact lịch sử **4 lần**, giữ lại các message gần nhất và tóm tắt phần cũ với độ dài giới hạn. Dù vậy, Advanced vẫn sinh **452 agent tokens**, nhiều hơn **374** của Baseline, vì câu trả lời có thêm dữ kiện. Như vậy, compact giúp giảm token đưa vào prompt; nó không nhất thiết làm phần câu trả lời ngắn hơn.

## Tăng trưởng bộ nhớ và giới hạn

Hồ sơ của Advanced tăng **345 bytes** trong Standard và **229 bytes** trong stress test. Baseline không tạo `User.md`, nên memory growth bằng **0 bytes**. Summary có giới hạn giúp tránh giữ nguyên toàn bộ lịch sử, nhưng khi compact, một số chi tiết cũ có thể bị mất. Bộ trích xuất cũng dựa trên quy tắc; nếu nhận nhầm một phát biểu thành correction rõ ràng, nó có thể ghi đè dữ kiện trước đó.

## Bonus: cập nhật khi có correction

Các fact được lưu theo từng field. Khi người dùng nói rõ thông tin mới, chẳng hạn chuyển nơi ở từ Huế sang Đà Nẵng, Advanced thay giá trị cũ thay vì giữ cả hai như thông tin hiện tại. Bộ trích xuất cũng bỏ qua câu hỏi, giả định và câu đùa như câu liên quan đến nghề `product manager`. Cách này giúp hồ sơ giữ dữ kiện mới nhất để trả lời ở thread khác; đổi lại, một correction sai nhưng được nói chắc chắn vẫn có thể làm hồ sơ sai. Có thể giảm rủi ro đó bằng bước xác nhận trước khi lưu những thay đổi quan trọng.

## Kiểm thử

Toàn bộ **32 kiểm thử** trong `src/` đều đạt. Các ca kiểm thử bao gồm lưu và cập nhật hồ sơ, recall giữa các thread, correction, giả định không chắc chắn, compact lặp lại và kiểm tra dữ liệu benchmark đầu vào.
