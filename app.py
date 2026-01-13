def make_signal_chart(df_plot, pred, proba, threshold, title):
    price_vals = df_plot["Último"].astype(float).values
    dates = df_plot["Data"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.70, 0.30],
    )

    # Preço
    fig.add_trace(
        go.Scatter(x=dates, y=price_vals, mode="lines", name="Preço"),
        row=1, col=1
    )

    # Marcadores (menores, sem poluir)
    fig.add_trace(
        go.Scatter(
            x=dates, y=np.where(pred == 1, price_vals, np.nan),
            mode="markers", name="ALTA",
            marker=dict(size=7, symbol="triangle-up")
        ),
        row=1, col=1
    )

    # Probabilidade (sem fill, simples)
    fig.add_trace(
        go.Scatter(
            x=dates, y=proba, mode="lines", name="P(ALTA)"
        ),
        row=2, col=1
    )

    # Threshold (sem annotation grande pra não “invadir”)
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_width=2,
        row=2, col=1
    )

    fig.update_layout(
        height=460,  # ✅ menor
        margin=dict(l=10, r=10, t=45, b=10),
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    fig.update_yaxes(title_text="Preço", row=1, col=1)
    fig.update_yaxes(title_text="P(ALTA)", range=[0, 1], row=2, col=1)

    # ✅ remove rangeslider (causava corte/overlap visual)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=2, col=1)

    return fig
