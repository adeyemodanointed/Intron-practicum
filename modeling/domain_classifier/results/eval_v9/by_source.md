# Validated texts by source

Top-label accuracy on the team-validated texts. v9 trains on FineWeb-2 pages (none of them containing a test text), so FineWeb-2-sourced test texts are reported apart.

| System | FineWeb-2 texts | Other validated texts | Validated `other` (FineWeb-2) | Validated `other` (rest) |
|---|---:|---:|---:|---:|
| base_v8 | 74.4% (n=215) | 72.1% (n=827) | 54.5% (n=66) | 57.3% (n=234) |
| tfidf6_v8 | 66.5% (n=215) | 70.0% (n=827) | 68.2% (n=66) | 63.7% (n=234) |
| ensemble_v8 | 71.2% (n=215) | 72.4% (n=827) | 63.6% (n=66) | 63.2% (n=234) |
| tfidf6_v8clean | 69.3% (n=215) | 70.5% (n=827) | 71.2% (n=66) | 67.1% (n=234) |
| tfidf6_v9 | 68.4% (n=215) | 69.3% (n=827) | 77.3% (n=66) | 69.7% (n=234) |
| tfidf6_v9clean | 67.4% (n=215) | 69.5% (n=827) | 77.3% (n=66) | 67.9% (n=234) |
| base_v9clean | 75.8% (n=215) | 71.8% (n=827) | 60.6% (n=66) | 61.1% (n=234) |
| ensemble_v9clean | 74.4% (n=215) | 72.1% (n=827) | 65.2% (n=66) | 62.0% (n=234) |
