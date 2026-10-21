"""Generate the reproducible synthetic filename dataset used by the project."""

import csv
from pathlib import Path


OUTPUT_PATH = Path(__file__).resolve().parent / "file_dataset.csv"

CATEGORIES = {
    "academic": {
        "stems": (
            "dbms_unit_3_lecture_notes",
            "calculus_integral_practice_set",
            "biology_cell_structure_lab_report",
            "physics_motion_experiment",
            "chemistry_periodic_table_revision",
            "statistics_probability_assignment",
            "computer_networks_exam_review",
            "operating_systems_process_scheduling",
            "machine_learning_feature_engineering",
            "data_structures_graph_algorithms",
            "microeconomics_market_analysis",
            "technical_writing_research_paper",
            "electrical_circuits_lab_manual",
            "organic_chemistry_reaction_notes",
            "linear_algebra_matrix_exercises",
            "database_systems_normalization_handout",
            "environmental_science_field_study",
            "anatomy_human_skeleton_study_guide",
            "software_testing_coursework",
            "digital_logic_design_assignment",
            "modern_history_source_analysis",
            "engineering_mechanics_problem_solutions",
            "language_linguistics_semantics_notes",
            "research_methods_literature_review",
            "astronomy_solar_system_project",
        ),
        "extensions": (".pdf", ".docx", ".pdf", ".pptx"),
    },
    "resume": {
        "stems": (
            "software_engineer_resume",
            "data_analyst_cv",
            "product_manager_resume",
            "graphic_designer_cv",
            "business_analyst_resume",
            "frontend_developer_cv",
            "project_coordinator_resume",
            "mechanical_engineer_cv",
            "marketing_specialist_resume",
            "cloud_architect_cv",
            "quality_assurance_resume",
            "ux_researcher_cv",
            "financial_accountant_resume",
            "python_developer_cv",
            "human_resources_resume",
            "network_administrator_cv",
            "research_assistant_resume",
            "technical_writer_cv",
            "operations_manager_resume",
            "cybersecurity_analyst_cv",
            "database_administrator_resume",
            "sales_executive_cv",
            "mobile_app_developer_resume",
            "supply_chain_planner_cv",
            "internship_application_resume",
        ),
        "extensions": (".pdf", ".docx", ".pdf", ".docx"),
    },
    "receipt": {
        "stems": (
            "grocery_market_receipt",
            "pharmacy_purchase_invoice",
            "coffee_shop_payment_receipt",
            "online_order_tax_invoice",
            "bookstore_sales_receipt",
            "fuel_station_transaction",
            "restaurant_dining_bill",
            "electronics_store_invoice",
            "clothing_shop_purchase_receipt",
            "utility_payment_confirmation",
            "hardware_store_sales_invoice",
            "hotel_booking_payment_receipt",
            "supermarket_checkout_slip",
            "medical_clinic_payment_receipt",
            "food_delivery_order_invoice",
            "parking_fee_payment_slip",
            "furniture_store_purchase_invoice",
            "mobile_recharge_transaction_receipt",
            "train_ticket_payment_confirmation",
            "pet_supply_store_sales_receipt",
            "subscription_billing_invoice",
            "bakery_counter_payment_receipt",
            "car_repair_service_invoice",
            "cinema_ticket_purchase_receipt",
            "office_supply_order_invoice",
        ),
        "extensions": (".pdf", ".jpg", ".pdf", ".png"),
    },
    "presentation": {
        "stems": (
            "database_normalization_slides",
            "quarterly_sales_review_deck",
            "renewable_energy_project_presentation",
            "product_launch_strategy_slides",
            "cybersecurity_awareness_deck",
            "customer_research_findings_presentation",
            "team_onboarding_slides",
            "cloud_migration_proposal_deck",
            "annual_budget_review_presentation",
            "mobile_app_design_walkthrough_slides",
            "market_competitor_analysis_deck",
            "sustainable_packaging_proposal_slides",
            "network_architecture_overview_deck",
            "project_status_update_presentation",
            "brand_guidelines_workshop_slides",
            "data_visualization_results_deck",
            "training_session_safety_presentation",
            "community_outreach_program_slides",
            "research_conference_talk_deck",
            "inventory_forecast_review_slides",
            "software_demo_walkthrough_presentation",
            "leadership_retreat_agenda_deck",
            "customer_support_process_slides",
            "business_continuity_plan_presentation",
            "internship_final_project_deck",
        ),
        "extensions": (".pptx", ".pdf", ".pptx", ".pptx"),
    },
    "image": {
        "stems": (
            "mountain_sunrise_landscape_photo",
            "garden_tomato_plant_closeup",
            "city_bridge_evening_snapshot",
            "family_beach_vacation_picture",
            "watercolor_flower_illustration",
            "orange_cat_window_portrait",
            "downtown_street_mural_photo",
            "blue_bicycle_product_shot",
            "forest_trail_autumn_scene",
            "homemade_bread_kitchen_photo",
            "historic_building_facade_image",
            "coastal_cliff_travel_picture",
            "hand_drawn_character_sketch",
            "night_sky_constellation_photo",
            "wooden_table_furniture_image",
            "wildflower_meadow_spring_photo",
            "ceramic_mug_studio_product_image",
            "snowy_park_path_snapshot",
            "abstract_shapes_digital_artwork",
            "fresh_pasta_dinner_picture",
            "lake_reflection_morning_landscape",
            "puppy_grass_playtime_photo",
            "architecture_floorplan_preview_image",
            "vintage_camera_collection_picture",
            "orange_sunset_harbor_landscape",
        ),
        "extensions": (".jpg", ".png", ".jpeg", ".webp"),
    },
    "other": {
        "stems": (
            "weekly_team_meeting_agenda",
            "home_internet_router_manual",
            "summer_road_trip_itinerary",
            "apartment_rental_agreement",
            "household_monthly_budget_tracker",
            "garden_watering_schedule",
            "wireless_headphones_warranty",
            "community_volunteer_contact_list",
            "birthday_party_guest_checklist",
            "personal_journal_entry",
            "home_office_inventory_sheet",
            "board_game_rulebook",
            "family_recipe_collection",
            "moving_day_task_checklist",
            "printer_driver_installation_guide",
            "neighborhood_event_flyer",
            "bike_maintenance_log",
            "phone_backup_archive",
            "language_learning_vocabulary_list",
            "household_recycling_calendar",
            "subscription_cancellation_letter",
            "weekend_camping_packing_list",
            "living_room_paint_color_samples",
            "freelance_client_call_summary",
            "desktop_wallpaper_collection",
        ),
        "extensions": (".pdf", ".txt", ".xlsx", ".docx", ".zip"),
    },
}

VARIANTS = (
    "draft",
    "final",
    "revised",
    "v2",
    "signed",
    "2025_01",
    "2025_06",
    "copy_02",
)


def generate_rows() -> list[dict[str, str]]:
    """Return balanced rows with varied filename suffixes and extensions."""
    rows = []
    for category, details in CATEGORIES.items():
        extensions = details["extensions"]
        for stem in details["stems"]:
            for index, variant in enumerate(VARIANTS):
                extension = extensions[index % len(extensions)]
                rows.append(
                    {
                        "file_name": f"{stem}_{variant}{extension}",
                        "extension": extension,
                        "category": category,
                    }
                )
    return rows


def write_dataset(output_path: Path = OUTPUT_PATH) -> int:
    """Write the dataset CSV and return the number of records written."""
    rows = generate_rows()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as dataset_file:
        writer = csv.DictWriter(
            dataset_file, fieldnames=("file_name", "extension", "category")
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    count = write_dataset()
    print(f"Wrote {count} synthetic filename records to {OUTPUT_PATH}")
