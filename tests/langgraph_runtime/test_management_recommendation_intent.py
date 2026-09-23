from src.autonomous.goal_builder import GoalBuilder


def test_recommendations_do_not_trigger_action_tracker_approval():
    goal = GoalBuilder().build(
        "Investigate July 2026 performance and provide prioritised management actions."
    )
    assert "management_action" not in {item.key for item in goal.criteria}


def test_explicit_action_tracker_request_still_requires_action_details():
    goal = GoalBuilder().build(
        "Create management action tracker for July 2026."
    )
    assert "management_action" in {item.key for item in goal.criteria}
