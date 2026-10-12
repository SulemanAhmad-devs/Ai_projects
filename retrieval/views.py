
from django.shortcuts import render

from .services import retrieve


def home(request):
    results = []
    question = ""
    error = ""

    if request.method == "POST":
        question = request.POST.get("question", "").strip()

        try:
            k = int(request.POST.get("k", 3))
            k = max(1, min(k, 10))

            threshold_value = request.POST.get(
                "threshold", ""
            ).strip()

            threshold = (
                float(threshold_value)
                if threshold_value
                else None
            )

            if threshold is not None and threshold < 0:
                raise ValueError("Threshold cannot be negative.")

            if question:
                results = retrieve(
                    query=question,
                    k=k,
                    threshold=threshold,
                )
            else:
                error = "Please enter a question."

        except ValueError:
            error = "Enter a valid K and a non-negative threshold."
        except Exception:
            # Keep internal exception details out of the webpage.
            error = (
                "Search failed. Check that your Chroma database "
                "exists and contains the expected collection."
            )

    return render(
        request,
        "retrieval/home.html",
        {
            "question": question,
            "results": results,
            "error": error,
        },
    )
