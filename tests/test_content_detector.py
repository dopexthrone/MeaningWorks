"""Tests for mother/content_detector.py — content-generation detection."""

import pytest
from mother.content_detector import ContentSignal, detect_content_request


class TestContentSignal:
    def test_default_not_detected(self):
        sig = ContentSignal()
        assert sig.detected is False
        assert sig.content_type == ""
        assert sig.directive == ""

    def test_frozen(self):
        sig = ContentSignal()
        with pytest.raises(AttributeError):
            sig.detected = True


class TestDetectContentRequest:
    """Core detection logic."""

    # --- Positive cases: original content types ---

    def test_blog_posts(self):
        sig = detect_content_request("write me some blog posts")
        assert sig.detected is True
        assert sig.content_type == "blog_post"

    def test_blog_posts_about_topic(self):
        sig = detect_content_request("write me some blog posts about AI")
        assert sig.detected is True
        assert sig.content_type == "blog_post"

    def test_documentation(self):
        sig = detect_content_request("create documentation for the API")
        assert sig.detected is True
        assert sig.content_type == "documentation"

    def test_letter(self):
        sig = detect_content_request("draft a letter to my landlord")
        assert sig.detected is True
        assert sig.content_type == "letter"

    def test_tutorial(self):
        sig = detect_content_request("write a detailed tutorial on React hooks")
        assert sig.detected is True
        assert sig.content_type == "tutorial"

    def test_proposal(self):
        sig = detect_content_request("draft a proposal for a new feature")
        assert sig.detected is True
        assert sig.content_type == "proposal"

    def test_email(self):
        sig = detect_content_request("compose an email to the team")
        assert sig.detected is True
        assert sig.content_type == "email"

    def test_essay(self):
        sig = detect_content_request("write an essay about climate change")
        assert sig.detected is True
        assert sig.content_type == "essay"

    def test_guide(self):
        sig = detect_content_request("create a comprehensive guide to Docker")
        assert sig.detected is True
        assert sig.content_type == "guide"

    def test_report(self):
        sig = detect_content_request("generate a report on Q4 metrics")
        assert sig.detected is True
        assert sig.content_type == "report"

    def test_readme(self):
        sig = detect_content_request("write a README for this project")
        assert sig.detected is True
        assert sig.content_type == "documentation"

    def test_summary(self):
        sig = detect_content_request("write a summary of the meeting")
        assert sig.detected is True
        assert sig.content_type == "summary"

    def test_outline(self):
        sig = detect_content_request("draft an outline for the presentation")
        assert sig.detected is True
        assert sig.content_type == "outline"

    def test_newsletter(self):
        sig = detect_content_request("write a newsletter for our subscribers")
        assert sig.detected is True
        assert sig.content_type == "newsletter"

    def test_white_paper(self):
        sig = detect_content_request("produce a white paper on blockchain")
        assert sig.detected is True
        assert sig.content_type == "white_paper"

    def test_press_release(self):
        sig = detect_content_request("draft a press release about the launch")
        assert sig.detected is True
        assert sig.content_type == "press_release"

    def test_case_study(self):
        sig = detect_content_request("write a case study about our client")
        assert sig.detected is True
        assert sig.content_type == "case_study"

    # --- Positive cases: business/professional content ---

    def test_business_plan(self):
        sig = detect_content_request("write a business plan for my startup")
        assert sig.detected is True
        assert sig.content_type == "business_plan"

    def test_pitch_deck(self):
        sig = detect_content_request("create a pitch deck for investors")
        assert sig.detected is True
        assert sig.content_type == "pitch_deck"

    def test_cover_letter(self):
        sig = detect_content_request("write a cover letter for the job application")
        assert sig.detected is True
        assert sig.content_type == "cover_letter"

    def test_job_description(self):
        sig = detect_content_request("draft a job description for a senior engineer")
        assert sig.detected is True
        assert sig.content_type == "job_description"

    def test_terms_of_service(self):
        sig = detect_content_request("write terms of service for our platform")
        assert sig.detected is True
        assert sig.content_type == "terms_of_service"

    def test_privacy_policy(self):
        sig = detect_content_request("draft a privacy policy")
        assert sig.detected is True
        assert sig.content_type == "privacy_policy"

    def test_contract(self):
        sig = detect_content_request("draft a contract for the freelancer")
        assert sig.detected is True
        assert sig.content_type == "contract"

    def test_resume(self):
        sig = detect_content_request("write a resume for a software engineer")
        assert sig.detected is True
        assert sig.content_type == "resume"

    def test_cv(self):
        sig = detect_content_request("draft a cv for me")
        assert sig.detected is True
        assert sig.content_type == "resume"

    def test_sop(self):
        sig = detect_content_request("write an sop for the deployment process")
        assert sig.detected is True
        assert sig.content_type == "sop"

    def test_rfp(self):
        sig = detect_content_request("draft an rfp for the vendor selection")
        assert sig.detected is True
        assert sig.content_type == "rfp"

    def test_faq(self):
        sig = detect_content_request("create a faq for our product")
        assert sig.detected is True
        assert sig.content_type == "faq"

    def test_runbook(self):
        sig = detect_content_request("write a runbook for incident response")
        assert sig.detected is True
        assert sig.content_type == "runbook"

    def test_playbook(self):
        sig = detect_content_request("create a playbook for onboarding")
        assert sig.detected is True
        assert sig.content_type == "playbook"

    def test_changelog(self):
        sig = detect_content_request("write a changelog for the release")
        assert sig.detected is True
        assert sig.content_type == "changelog"

    def test_release_notes(self):
        sig = detect_content_request("draft release notes for v2.0")
        assert sig.detected is True
        assert sig.content_type == "release_notes"

    def test_investor_update(self):
        sig = detect_content_request("write an investor update for Q1")
        assert sig.detected is True
        assert sig.content_type == "investor_update"

    def test_strategy_document(self):
        sig = detect_content_request("draft a strategy doc for the product")
        assert sig.detected is True
        assert sig.content_type == "strategy_document"

    def test_mission_statement(self):
        sig = detect_content_request("write a mission statement")
        assert sig.detected is True
        assert sig.content_type == "mission_statement"

    # --- Positive cases: social/marketing content ---

    def test_social_media_post(self):
        sig = detect_content_request("write some social media posts about our launch")
        assert sig.detected is True
        assert sig.content_type == "social_media_post"

    def test_linkedin_post(self):
        sig = detect_content_request("draft a linkedin post about my new role")
        assert sig.detected is True
        assert sig.content_type == "social_media_post"

    def test_tweet(self):
        sig = detect_content_request("write a tweet announcing the feature")
        assert sig.detected is True
        assert sig.content_type == "tweet"

    def test_ad_copy(self):
        sig = detect_content_request("write ad copy for the campaign")
        assert sig.detected is True
        assert sig.content_type == "ad_copy"

    def test_tagline(self):
        sig = detect_content_request("create a tagline for the brand")
        assert sig.detected is True
        assert sig.content_type == "tagline"

    def test_slogan(self):
        sig = detect_content_request("write a slogan for the product")
        assert sig.detected is True
        assert sig.content_type == "slogan"

    def test_testimonial(self):
        sig = detect_content_request("draft a testimonial for the website")
        assert sig.detected is True
        assert sig.content_type == "testimonial"

    def test_caption(self):
        sig = detect_content_request("write a caption for this photo")
        assert sig.detected is True
        assert sig.content_type == "caption"

    def test_landing_page(self):
        sig = detect_content_request("write landing page copy for the product")
        assert sig.detected is True
        assert sig.content_type == "landing_page"

    # --- Positive cases: verb coverage ---

    def test_verb_make(self):
        sig = detect_content_request("make me a business plan")
        assert sig.detected is True
        assert sig.content_type == "business_plan"

    def test_verb_develop(self):
        sig = detect_content_request("develop a proposal for the board")
        assert sig.detected is True
        assert sig.content_type == "proposal"

    def test_verb_formulate(self):
        sig = detect_content_request("formulate a strategy doc")
        assert sig.detected is True
        assert sig.content_type == "strategy_document"

    def test_verb_design(self):
        sig = detect_content_request("design an onboarding guide for new hires")
        assert sig.detected is True
        assert sig.content_type == "onboarding_document"

    def test_verb_put_together(self):
        """BUG #3 fix — 'put together' is a common writing phrase."""
        sig = detect_content_request("put together a report for the meeting")
        assert sig.detected is True
        assert sig.content_type == "report"

    def test_verb_put_together_proposal(self):
        sig = detect_content_request("put together a proposal")
        assert sig.detected is True
        assert sig.content_type == "proposal"

    # --- Positive cases: quantity/length signals ---

    def test_couple_signal(self):
        sig = detect_content_request("I need a couple blog posts")
        assert sig.detected is True

    def test_batch_signal(self):
        sig = detect_content_request("give me a batch of social media posts")
        assert sig.detected is True

    def test_series_signal(self):
        sig = detect_content_request("I want a series of articles on AI")
        assert sig.detected is True

    def test_deep_dive_signal(self):
        sig = detect_content_request("I want a deep dive article on Rust")
        assert sig.detected is True

    def test_quantity_signal_triggers(self):
        """Quantity/length signal + content noun, no writing verb."""
        sig = detect_content_request("I need some blog posts about marketing")
        assert sig.detected is True
        assert sig.content_type == "blog_post"

    def test_length_signal_triggers(self):
        sig = detect_content_request("I need a detailed guide to Python")
        assert sig.detected is True
        assert sig.content_type == "guide"

    def test_comprehensive_signal(self):
        sig = detect_content_request("give me a comprehensive report")
        assert sig.detected is True

    def test_flesh_out_signal(self):
        """BUG #5 fix — uninflected 'flesh out' now detected."""
        sig = detect_content_request("flesh out this proposal")
        assert sig.detected is True
        assert sig.content_type == "proposal"

    def test_fleshed_out_signal(self):
        sig = detect_content_request("I want a fleshed out guide")
        assert sig.detected is True

    # --- Positive cases: content noun + code topic (correct detection) ---

    def test_description_of_tool(self):
        """User wants written content about a tool — should detect."""
        sig = detect_content_request("write a description of the tool")
        assert sig.detected is True
        assert sig.content_type == "description"

    def test_guide_for_api(self):
        """User wants a written guide about an API — should detect."""
        sig = detect_content_request("write a guide for the API")
        assert sig.detected is True
        assert sig.content_type == "guide"

    def test_tutorial_about_functions(self):
        sig = detect_content_request("write a tutorial about functions")
        assert sig.detected is True
        assert sig.content_type == "tutorial"

    def test_description_of_pipeline(self):
        sig = detect_content_request("write a description of the pipeline")
        assert sig.detected is True
        assert sig.content_type == "description"

    # --- Positive cases: ambiguous nouns in content context ---

    def test_script_for_play(self):
        """'script' in theater context — should detect."""
        sig = detect_content_request("write a script for the play")
        assert sig.detected is True
        assert sig.content_type == "script"

    def test_thread_about_ai(self):
        """'thread' in social media context — should detect."""
        sig = detect_content_request("write a thread about AI")
        assert sig.detected is True
        assert sig.content_type == "thread"

    def test_review_of_restaurant(self):
        """'review' in written content context — should detect."""
        sig = detect_content_request("write a review of the restaurant")
        assert sig.detected is True
        assert sig.content_type == "review"

    # --- Negative cases: should NOT detect ---

    def test_casual_greeting(self):
        sig = detect_content_request("how are you?")
        assert sig.detected is False

    def test_casual_question(self):
        sig = detect_content_request("what's the weather like?")
        assert sig.detected is False

    def test_code_request_function(self):
        sig = detect_content_request("write a Python function")
        assert sig.detected is False

    def test_code_request_class(self):
        sig = detect_content_request("create a class for user auth")
        assert sig.detected is False

    def test_code_request_fix(self):
        sig = detect_content_request("fix the bug in the login")
        assert sig.detected is False

    def test_build_request(self):
        sig = detect_content_request("build me a booking system")
        assert sig.detected is False

    def test_write_alone(self):
        sig = detect_content_request("write")
        assert sig.detected is False

    def test_blog_alone(self):
        sig = detect_content_request("blog")
        assert sig.detected is False

    def test_empty_string(self):
        sig = detect_content_request("")
        assert sig.detected is False

    def test_none_like(self):
        sig = detect_content_request("   ")
        assert sig.detected is False

    def test_code_write_module(self):
        sig = detect_content_request("write a module for data processing")
        assert sig.detected is False

    def test_code_write_test(self):
        sig = detect_content_request("write tests for the API")
        assert sig.detected is False

    def test_code_create_endpoint(self):
        sig = detect_content_request("create an endpoint for users")
        assert sig.detected is False

    def test_code_script_python(self):
        sig = detect_content_request("write a python script to parse CSV")
        assert sig.detected is False

    def test_code_make_app(self):
        sig = detect_content_request("make me an app")
        assert sig.detected is False

    def test_code_design_system(self):
        sig = detect_content_request("design a database schema")
        assert sig.detected is False

    def test_code_develop_feature(self):
        sig = detect_content_request("develop a new feature for the dashboard")
        assert sig.detected is False

    def test_code_create_website(self):
        sig = detect_content_request("create a website for my business")
        assert sig.detected is False

    def test_code_make_bot(self):
        sig = detect_content_request("make a chatbot for customer support")
        assert sig.detected is False

    def test_code_create_dashboard(self):
        sig = detect_content_request("create a dashboard for analytics")
        assert sig.detected is False

    # --- Negative: ambiguous nouns in code context ---

    def test_thread_pool(self):
        """BUG #1 fix — 'create a thread pool' is code, not content."""
        sig = detect_content_request("create a thread pool")
        assert sig.detected is False

    def test_thread_safe(self):
        sig = detect_content_request("make a thread safe wrapper")
        assert sig.detected is False

    def test_thread_executor(self):
        sig = detect_content_request("create a thread executor")
        assert sig.detected is False

    def test_code_review(self):
        """BUG #2 fix — 'create a code review' is dev workflow, not content."""
        sig = detect_content_request("create a code review")
        assert sig.detected is False

    def test_pr_review(self):
        sig = detect_content_request("write a pr review")
        assert sig.detected is False

    def test_merge_review(self):
        sig = detect_content_request("draft a merge review")
        assert sig.detected is False

    def test_script_python(self):
        sig = detect_content_request("write a python script")
        assert sig.detected is False

    def test_script_bash(self):
        sig = detect_content_request("create a bash script")
        assert sig.detected is False

    def test_script_typescript(self):
        sig = detect_content_request("write a typescript script")
        assert sig.detected is False

    # --- Negative: substring collision fixes (BUGs #6, #7, #8) ---

    def test_someone_wrote_guide(self):
        """BUG #7 fix — 'some' must not match inside 'someone'."""
        sig = detect_content_request("someone wrote a guide")
        assert sig.detected is False

    def test_something_about_report(self):
        sig = detect_content_request("something about the report")
        assert sig.detected is False

    def test_somehow_email_lost(self):
        sig = detect_content_request("somehow the email got lost")
        assert sig.detected is False

    def test_somebody_needs_brief(self):
        sig = detect_content_request("somebody needs the brief")
        assert sig.detected is False

    def test_completely_forgot_guide(self):
        """BUG #6 fix — 'complete' must not match inside 'completely'."""
        sig = detect_content_request("I completely forgot about the guide")
        assert sig.detected is False

    def test_completely_agree_proposal(self):
        sig = detect_content_request("I completely agree with the proposal")
        assert sig.detected is False

    def test_along_the_lines_proposal(self):
        """BUG #8 fix — 'long' must not match inside 'along'."""
        sig = detect_content_request("I've been thinking along the lines of a proposal")
        assert sig.detected is False

    def test_belong_to_guide(self):
        sig = detect_content_request("these notes belong to the guide")
        assert sig.detected is False

    def test_fullstack_developer(self):
        """'full' must not match inside 'fullstack'."""
        sig = detect_content_request("I need a fullstack developer to help with docs")
        assert sig.detected is False

    # --- Positive: quantity signals still work at word boundaries ---

    def test_some_as_whole_word(self):
        """'some' as a whole word should still detect."""
        sig = detect_content_request("I need some blog posts")
        assert sig.detected is True

    def test_long_as_whole_word(self):
        sig = detect_content_request("write a long article about startups")
        assert sig.detected is True

    def test_complete_as_whole_word(self):
        sig = detect_content_request("write a complete guide to Python")
        assert sig.detected is True

    def test_full_as_whole_word(self):
        sig = detect_content_request("write a full report on the findings")
        assert sig.detected is True

    # --- Directive format ---

    def test_directive_contains_marker(self):
        sig = detect_content_request("write me some blog posts")
        assert "[Response Mode: Content Generation]" in sig.directive

    def test_directive_contains_content_type(self):
        sig = detect_content_request("draft a proposal")
        assert "proposal" in sig.directive

    def test_directive_instructs_full_output(self):
        sig = detect_content_request("write a tutorial")
        assert "Produce the actual content" in sig.directive
        assert "do not just acknowledge" in sig.directive

    # --- Case insensitivity ---

    def test_case_insensitive_verb(self):
        sig = detect_content_request("WRITE me a blog post")
        assert sig.detected is True

    def test_case_insensitive_noun(self):
        sig = detect_content_request("draft a PROPOSAL")
        assert sig.detected is True

    def test_mixed_case(self):
        sig = detect_content_request("Create A Tutorial on TypeScript")
        assert sig.detected is True

    # --- Substring collision prevention (noun ordering) ---

    def test_newsletter_not_letter(self):
        sig = detect_content_request("write a newsletter")
        assert sig.content_type == "newsletter"

    def test_cover_letter_not_letter(self):
        sig = detect_content_request("write a cover letter")
        assert sig.content_type == "cover_letter"

    def test_blog_posts_plural(self):
        sig = detect_content_request("write blog posts about startups")
        assert sig.content_type == "blog_post"

    def test_press_releases_plural(self):
        sig = detect_content_request("draft press releases for the events")
        assert sig.content_type == "press_release"

    # --- Conversational phrasing that should work ---

    def test_can_you_write(self):
        sig = detect_content_request("can you write me a few blog posts?")
        assert sig.detected is True

    def test_id_like_detailed(self):
        sig = detect_content_request("I'd like a detailed proposal")
        assert sig.detected is True

    def test_help_me_write(self):
        sig = detect_content_request("help me write a business plan")
        assert sig.detected is True

    def test_whip_up_via_quantity(self):
        """'whip up' not a verb, but 'some' quantity signal carries it."""
        sig = detect_content_request("whip up some blog posts")
        assert sig.detected is True
