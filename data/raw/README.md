# Data Placeholder

Place your raw customer support ticket dataset in this directory.

Default configuration expects the CFPB consumer complaints export:
- File name: `consumer_complaints.json`
- Text field: `_source.complaint_what_happened`
- Label field: `_source.product`

If you supply a different dataset, adjust `config/default.yaml` to point at the new file
and update the expected text/label columns accordingly.
