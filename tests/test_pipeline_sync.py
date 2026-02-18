from __future__ import annotations

from intelligence.pipeline import _prune_off_topic_results, _sync_video_brief_with_flow


def test_sync_video_brief_with_flow_expands_to_slide_count():
    video_brief = {
        "title": "Brief",
        "segments": [
            {
                "title": "Only one segment",
                "content": "segment content",
                "talking_points": ["point"],
                "duration_sec": 15,
            }
        ],
    }
    flow = {
        "slide_timeline": [
            {"slide_index": 1, "title": "Slide 1", "start_sec": 0, "duration_sec": 18},
            {"slide_index": 2, "title": "Slide 2", "start_sec": 18, "duration_sec": 22},
            {"slide_index": 3, "title": "Slide 3", "start_sec": 40, "duration_sec": 20},
        ]
    }

    synced = _sync_video_brief_with_flow(video_brief=video_brief, flow=flow)
    assert synced is not None
    segments = synced.get("segments") or []
    assert len(segments) == 3
    assert segments[0]["start_sec"] == 0
    assert segments[1]["start_sec"] == 18
    assert segments[2]["duration_sec"] == 20
    assert synced.get("duration_estimate") == "1m 00s"


def test_prune_off_topic_results_prefers_matching_version_markers():
    results = [
        {
            "id": "r1",
            "source": "arxiv",
            "title": "Kimi Chat 1.5 Technical Report",
            "content": "Older generation details.",
        },
        {
            "id": "r2",
            "source": "arxiv",
            "title": "Kimi Chat 2.5 Technical Report",
            "content": "Latest architecture details.",
        },
        {
            "id": "r3",
            "source": "github",
            "title": "Kimi Chat runtime",
            "content": "No explicit version marker",
        },
    ]

    filtered, summary = _prune_off_topic_results(
        search_results=results,
        topic="Kimi Chat 2.5",
        query_rewrites=["Kimi 2.5 latest"],
        min_keep=2,
    )

    kept_ids = {item["id"] for item in filtered}
    assert "r2" in kept_ids
    assert "r1" not in kept_ids
    assert summary.get("version_markers") == ["2.5"]


def test_prune_off_topic_results_keeps_strict_version_filter_when_candidate_pool_small():
    results = [
        {
            "id": "old_1_5",
            "source": "arxiv",
            "title": "Kimi Chat 1.5 Technical Report",
            "content": "Older generation details.",
        },
        {
            "id": "target_2_5",
            "source": "arxiv",
            "title": "Kimi Chat 2.5 Technical Report",
            "content": "Latest architecture details.",
        },
    ]

    filtered, summary = _prune_off_topic_results(
        search_results=results,
        topic="Kimi Chat 2.5",
        query_rewrites=["Kimi 2.5 latest"],
        min_keep=10,
    )

    kept_ids = {item["id"] for item in filtered}
    assert "target_2_5" in kept_ids
    assert "old_1_5" not in kept_ids
    assert summary.get("version_strict_applied") is True
