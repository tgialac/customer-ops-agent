# Workflow 3: Google Play app refund

This workflow covers a customer asking how to request a refund for an app
purchased through Google Play. It is not a merchant refund or cashback flow.

## Source-backed behavior

The official MoMo FAQ instructs the customer to:

1. Open Google Play and choose **Tài khoản**.
2. Open **Lịch sử đơn đặt hàng**.
3. Select the app and choose **Báo cáo sự cố**.
4. Receive the result in the Google Play account email and in the MoMo app.

The source says processing time depends on Google Play. The agent therefore
does not invent a deadline, decide eligibility, or promise that MoMo will
approve the refund.

Source: [MoMo FAQ: refund an app purchase](https://www.momo.vn/hoi-dap/toi-muon-hoan-tien-da-giao-dich-cho-ung-dung-da-mua)

The source-backed regression suite is
`data/golden/google_play_refund_v1.jsonl`.
