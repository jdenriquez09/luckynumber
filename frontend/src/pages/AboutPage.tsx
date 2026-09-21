export default function AboutPage() {
  return (
    <div className="max-w-2xl">
      <h1 className="text-2xl font-bold mb-6">About This Site</h1>

      <section className="mb-6">
        <h2 className="text-lg font-semibold mb-2">What this is</h2>
        <p className="text-slate-600 text-sm leading-relaxed">
          This platform collects historical PCSO lotto draw results and analyses them
          statistically to test whether they are consistent with a fair, random process.
          It is an educational and analytical tool — not a gambling aid.
        </p>
      </section>

      <section className="mb-6">
        <h2 className="text-lg font-semibold mb-2">What this is NOT</h2>
        <div className="bg-amber-50 border border-amber-200 rounded p-4 text-sm text-amber-900">
          <ul className="list-disc list-inside space-y-1">
            <li>This site does <strong>not</strong> predict future draw results.</li>
            <li>There are no "hot", "cold", or "overdue" numbers — lotto draws are designed to be independent.</li>
            <li>A number that hasn't appeared recently is <strong>not more likely</strong> to appear next.</li>
            <li>No number combination has better odds than any other.</li>
          </ul>
        </div>
      </section>

      <section className="mb-6">
        <h2 className="text-lg font-semibold mb-2">Statistical methods</h2>
        <ul className="text-sm text-slate-600 list-disc list-inside space-y-1">
          <li>Chi-square goodness-of-fit for number frequency</li>
          <li>Distribution tests: draw sum, odd/even split, consecutive pairs</li>
          <li>Pair co-occurrence with Benjamini-Hochberg FDR correction</li>
          <li>Gap analysis vs geometric expectation</li>
          <li>Wald-Wolfowitz runs test, lag-1 autocorrelation</li>
          <li>Monte Carlo simulation (10,000 fair histories)</li>
          <li>ANOVA for weekday and monthly effects</li>
        </ul>
      </section>

      <section className="mb-6">
        <h2 className="text-lg font-semibold mb-2">Data sources</h2>
        <p className="text-sm text-slate-600">
          Primary: <a href="https://www.lottopcso.com" className="text-blue-600 hover:underline" target="_blank" rel="noreferrer">lottopcso.com</a>.
          Cross-checked against pcso.gov.ph. Data coverage may be incomplete;
          check the per-game completeness report for actual date ranges.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-2">Limitations</h2>
        <p className="text-sm text-slate-600">
          Statistical significance with many simultaneous tests is expected by chance alone.
          All findings include p-values, sample sizes, and multiple-comparison corrections.
          A significant result does not mean draws are unfair; it warrants closer inspection.
        </p>
      </section>
    </div>
  );
}
