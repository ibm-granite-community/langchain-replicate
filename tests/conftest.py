# SPDX-License-Identifier: MIT

import os
from typing import Any

import pytest
from assertpy import assert_that
from pydantic.types import SecretStr


@pytest.fixture
def replicate_api_token(monkeypatch) -> SecretStr:
    """Return the api token from the env. We also remove it from the env."""
    api_token = os.getenv("REPLICATE_API_TOKEN")
    assert_that(api_token).is_not_none().is_not_empty()
    monkeypatch.delenv("REPLICATE_API_TOKEN")
    return SecretStr(api_token)  # type: ignore


@pytest.fixture
def documents() -> list[dict[str, Any]]:
    # pylint: disable=line-too-long
    doc_list: list[dict[str, Any]] = [
        {
            "text": """Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you're at it, pass the Disclose Act so Americans can know who is funding our elections.
Tonight, I'd like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service.
One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.
And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation's top legal minds, who will continue Justice Breyer's legacy of excellence.
A former top litigator in private practice. A former federal public defender. And from a family of public school educators and police officers. A consensus builder. Since she's been nominated, she's received a broad range of support—from the Fraternal Order of Police to former judges appointed by Democrats and Republicans.
And if we are to advance liberty and justice, we need to secure the Border and fix the immigration system.
We can do both. At our border, we've installed new technology like cutting-edge scanners to better detect drug smuggling.
We've set up joint patrols with Mexico and Guatemala to catch more human traffickers.
We're putting in place dedicated immigration judges so families fleeing persecution and violence can have their cases heard faster.
We're securing commitments and supporting partners in South and Central America to host more refugees and secure their own borders.
We can do all this while keeping lit the torch of liberty that has led generations of immigrants to this land—my forefathers and so many of yours.
Provide a pathway to citizenship for Dreamers, those on temporary status, farm workers, and essential workers.
Revise our laws so businesses have the workers they need and families don't wait decades to reunite.
It's not only the right thing to do—it's the economically smart thing to do.
That's why immigration reform is supported by everyone from labor unions to religious leaders to the U.S. Chamber of Commerce.""",  # noqa: E501
            "doc_id": "15",
        },
        {
            "text": """That's why I've proposed closing loopholes so the very wealthy don't pay a lower tax rate than a teacher or a firefighter.
So that's my plan. It will grow the economy and lower costs for families.
So what are we waiting for? Let's get this done. And while you're at it, confirm my nominees to the Federal Reserve, which plays a critical role in fighting inflation.
My plan will not only lower costs to give families a fair shot, it will lower the deficit.
The previous Administration not only ballooned the deficit with tax cuts for the very wealthy and corporations, it undermined the watchdogs whose job was to keep pandemic relief funds from being wasted.
But in my administration, the watchdogs have been welcomed back.
We're going after the criminals who stole billions in relief money meant for small businesses and millions of Americans.
And tonight, I'm announcing that the Justice Department will name a chief prosecutor for pandemic fraud.
By the end of this year, the deficit will be down to less than half what it was before I took office.
The only president ever to cut the deficit by more than one trillion dollars in a single year.
Lowering your costs also means demanding more competition.
I'm a capitalist, but capitalism without competition isn't capitalism.
It's exploitation—and it drives up prices.
When corporations don't have to compete, their profits go up, your prices go up, and small businesses and family farmers and ranchers go under.
We see it happening with ocean carriers moving goods in and out of America.
During the pandemic, these foreign-owned companies raised prices by as much as 1,000% and made record profits.
Tonight, I'm announcing a crackdown on these companies overcharging American businesses and consumers.
And as Wall Street firms take over more nursing homes, quality in those homes has gone down and costs have gone up.
That ends on my watch.
Medicare is going to set higher standards for nursing homes and make sure your loved ones get the care they deserve and expect.""",  # noqa: E501
            "doc_id": "10",
        },
        {
            "text": """Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.
Last year COVID-19 kept us apart. This year we are finally together again.
Tonight, we meet as Democrats Republicans and Independents. But most importantly as Americans.
With a duty to one another to the American people to the Constitution.
And with an unwavering resolve that freedom will always triumph over tyranny.
Six days ago, Russia's Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated.
He thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined.
He met the Ukrainian people.
From President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world.
Groups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.
In this struggle as President Zelenskyy said in his speech to the European Parliament “Light will win over darkness.” The Ukrainian Ambassador to the United States is here tonight.
Let each of us here tonight in this Chamber send an unmistakable signal to Ukraine and to the world.
Please rise if you are able and show that, Yes, we the United States of America stand with the Ukrainian people.
Throughout our history we've learned this lesson when dictators do not pay a price for their aggression they cause more chaos.
They keep moving.
And the costs and the threats to America and the world keep rising.
That's why the NATO Alliance was created to secure peace and stability in Europe after World War 2.
The United States is a member along with 29 other nations.
It matters. American diplomacy matters. American resolve matters.
Putin's latest attack on Ukraine was premeditated and unprovoked.
He rejected repeated efforts at diplomacy.
He thought the West and NATO wouldn't respond. And he thought he could divide us at home. Putin was wrong. We were ready.  Here is what we did.
We prepared extensively and carefully.""",  # noqa: E501
            "doc_id": "1",
        },
        {
            "text": """It's time for Americans to get back to work and fill our great downtowns again.  People working from home can feel safe to begin to return to the office.
We're doing that here in the federal government. The vast majority of federal workers will once again work in person.
Our schools are open. Let's keep it that way. Our kids need to be in school.
And with 75% of adult Americans fully vaccinated and hospitalizations down by 77%, most Americans can remove their masks, return to work, stay in the classroom, and move forward safely.
We achieved this because we provided free vaccines, treatments, tests, and masks.
Of course, continuing this costs money.
I will soon send Congress a request.
The vast majority of Americans have used these tools and may want to again, so I expect Congress to pass it quickly.
Fourth, we will continue vaccinating the world.
We've sent 475 Million vaccine doses to 112 countries, more than any other nation.
And we won't stop.
We have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life.
Let's use this moment to reset. Let's stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.
Let's stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.
We can't change how divided we've been. But we can change how we move forward—on COVID-19 and other issues we must face together.
I recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera.
They were responding to a 9-1-1 call when a man shot and killed them with a stolen gun.
Officer Mora was 27 years old.
Officer Rivera was 22.
Both Dominican Americans who'd grown up on the same streets they later chose to patrol as police officers.""",  # noqa: E501
            "doc_id": 13,
        },
    ]
    return doc_list


@pytest.fixture
def tools() -> list[dict[str, Any]]:
    tool_list: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"description": "The city, e.g. San Francisco, CA", "type": "string"},
                        "unit": {"enum": ["celsius", "fahrenheit"], "type": "string"},
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Retrieves the lowest and highest stock prices for a given ticker and date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": 'The stock ticker symbol, e.g., "IBM".'},
                        "date": {"type": "string", "description": 'The date in "YYYY-MM-DD" format for which you want to get stock prices.'},
                    },
                    "required": ["ticker", "date"],
                },
            },
        },
    ]
    return tool_list


@pytest.fixture
def embed_texts() -> list[str]:
    texts = ["What is Pi?", "Cats are mammals", "Snakes are reptiles"]
    return texts
