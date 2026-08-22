---
document_id: momo-developer-refund-api-2026-08-22
title: Hoàn tiền giao dịch qua MoMo Payment API
source_url: https://developers.momo.vn/v3/vi/docs/payment/api/payment-api/refund/
source_kind: official_developer_docs
audience: merchant
topic: merchant_refund
checked_at: 2026-08-22
status: active
keywords: merchant, đối tác, API, refund, hoàn tiền, giao dịch thành công, hoàn một phần, hoàn toàn phần
---

Phạm vi: tài liệu tích hợp dành cho merchant/đối tác gọi MoMo Payment API; không phải hướng dẫn tự phục vụ cho khách hàng cuối.

API refund được dùng cho giao dịch thành công có mã trạng thái 0. MoMo hỗ trợ hoàn một phần khi số tiền hoàn nhỏ hơn số tiền đã thanh toán và hoàn toàn phần khi số tiền hoàn bằng số tiền đã thanh toán. Tài liệu cũng khuyến nghị timeout tối thiểu 30 giây khi gọi API.

Agent hỗ trợ khách hàng không được dùng tài liệu này để hứa rằng một giao dịch cụ thể sẽ được hoàn. Cần kiểm tra đúng quy trình của merchant và chuyển yêu cầu cho merchant hoặc bộ phận hỗ trợ khi cần.
